import os
import pandas as pd

from flask import Flask, request, render_template
from pathlib import Path
from mmdet.apis import DetInferencer
from PIL import Image
from pyarrow.feather import read_feather
from scipy.stats import percentileofscore


# # set working directory
wdir = Path(os.getcwd())
os.chdir(wdir)

# init hyperparameters
#     image_threshold (float): threshold for the object confidence in the user-uploaded image
#     context (str): context of the image, either "kickstarter" or "general"
image_threshold = 0.1
context = "general"

# Initialize the DetInferencer
inferencer = DetInferencer(
    # Deformable-DETR (DETR is faster)
    model=str(
        wdir
        / "data/V3Det/checkpoints/configs/v3det/deformable-detr-refine-twostage_swin_16xb2_sample1e-3_v3det_50e.py"
    ),
    weights=str(wdir / "data/V3Det/checkpoints/Deformable_DETR_V3Det_SwinB.pth"),
    device="cpu",
)


def get_metrics_dist_v3d():
    """get the metrics distribution of V3D
    Return
        metrics_dist_v3d: a DataFrame with the following columns:
            - pid (int): the project id
            - mni (float): the mni of the project
            - readability (float): the readability of the project
            - uniqueness (float): the uniqueness of the project
    """
    metrics_dist_v3d = read_feather("data/metrics_dist_p50_allv3d.feather")
    metrics_dist_v3d["uniqueness"] = metrics_dist_v3d["uniqueness"] * 1000

    return metrics_dist_v3d


metrics_dist_v3d = get_metrics_dist_v3d()


def get_obj_freq(score_threshold):
    # load the object-level frequency table (Kick- and V3D-context)
    freq_v3d = read_feather(
        # for v3d, we only use one threahold (0.5)
        "data/freq_v3d_p50.feather",
        columns=["label", "freq"],
    )
    freq_v3d = freq_v3d.rename(columns={"freq": "freq_v3d"})

    freq = freq_v3d

    return freq


obj_freq = get_obj_freq(image_threshold)


def get_obj_mni(score_threshold):
    """get the object-level MNI
    Returns:
        obj_mni: a DataFrame that contains the mni of each object

    """

    # load MNI based on V3D (for V3D we only use one threshold, 0.5)
    obj_mni_k100_v3d = read_feather("data/obj_mni_v3d_k100_p50.feather")

    # rename columns in obj_mni_v3d
    obj_mni_k100_v3d = obj_mni_k100_v3d.rename(columns={"mni": "mni_k100_v3d"})

    # merge all MNI datasets
    obj_mni = obj_mni_k100_v3d

    return obj_mni


obj_mni = get_obj_mni(image_threshold)


def get_objects(img_path):
    """Identify the objects in the image
    Args:
        img_path: the path to the image

    Returns:
        a DataFrame with the following columns:
        - label (int): the label of the object
        - score: the confidence score of the object
        - size_ratio: the ratio of the size of the object to the size of the image
    """

    # Get the objects in the image
    objects = inferencer(str(img_path), show=False)["predictions"][0]

    # get size of each object
    obj_size = [w * h for x, y, w, h in objects["bboxes"]]

    # get the image size
    with Image.open(img_path) as img:
        w, h = img.size
        img_size = w * h

    # get the ratio of each object
    obj_size_ratio = [x / img_size for x in obj_size]

    # collect the results into a dataframe
    df = pd.DataFrame(
        {
            "name": img_path.stem,
            "label": objects["labels"],
            "score": objects["scores"],
            "size_ratio": obj_size_ratio,
        }
    )

    # the output label starts from 0, but V3D label starts from 1
    # so we add 1 to the label
    df["label"] = df.label.astype(int) + int(1)

    return df


def get_uniqueness(obj_df, freq):
    """Calculate uniqueness of the image
    Args:
        obj_df (DataFrame): a DataFrame with the following columns:
            - label (int): the label of the object
            - score: the confidence score of the object
            - size_ratio: the ratio of the size of the object to the size of the image

        freq: a DataFrame that contains the frequency score of every object

    Returns:
        a DataFrame with the following columns:
            - name (str): name of the image
            - freq_v3d (float): frequency of the object in the general context
    """

    # we only keep objects with score >= image_threshold
    obj_df = obj_df.loc[obj_df.score >= image_threshold]

    # compute the freq of each project by aggregating the freq of each object in the project
    uniqueness = (
        obj_df.merge(freq, on="label", how="inner")
        .groupby(["name"])
        .agg({"freq_v3d": "mean"})
        .rename(columns={"freq_v3d": "uniqueness"})
        .reset_index()
    )

    # multiply uniqueness by 1000
    uniqueness["uniqueness"] = uniqueness["uniqueness"] * 1000

    return uniqueness


def get_readability(obj_df):
    """Calculate gunning fog index of the image
    Args:
        obj_df (DataFrame): a DataFrame with the following columns:
            - label (int): the label of the object
            - score: the confidence score of the object
            - size_ratio: the ratio of the size of the object to the size of the image

    Returns:
        a DataFrame with the following columns:
            - name (str): name of the image
            - readability (float): gunning fog index of the image
    """

    # get object number and object size
    def _readability(group):
        # compute object number
        obj_num = len(group)

        # compute object size
        obj_size_lt_10 = sum(group.size_ratio <= 0.1)

        # get gunning fog index
        readability = 0.4 * (obj_num + 100 * obj_size_lt_10 / obj_num)

        # return
        return pd.Series(readability, index=["readability"])

    out = (
        obj_df.loc[obj_df.score >= image_threshold]
        .groupby(["name"])
        .apply(_readability)
        .reset_index()
    )

    return out


def get_mni(obj_df, obj_mni):
    """Calculate mni (concreteness) of the image
    Args:
        obj_df (DataFrame): a DataFrame with the following columns:
            - label (int): the label of the object
            - score: the confidence score of the object
            - size_ratio: the ratio of the size of the object to the size of the image

    Returns:
        a DataFrame with the following columns:
            - name (str): name of the image
            - mni (float): mni of the image
    """

    # --- compute project-level MNI --- #
    obj_df = obj_df.loc[obj_df.score >= image_threshold]

    # compute project-level MNI
    out = (
        obj_df.merge(obj_mni, left_on="label", right_on="obj", how="inner")
        .groupby(["name"])
        .agg({"mni_k100_v3d": "mean"})
        .rename(columns={"mni_k100_v3d": "mni"})
        .reset_index()
    )

    return out


def get_metrics(uniqueness, readability, mni):
    """Combine all the metrics (uniqueness, gunning-fog, mni) into a dictionary
    Args:
        uniqueness: a DataFrame of uniqueness
        readability: a DataFrame of readability
        mni: a DataFrame of mni

    Returns:
        a DataFrame with all the metrics
    """

    # combine all the metrics
    metrics = (
        uniqueness.merge(readability, on="name", how="inner")
        .merge(mni, on="name", how="inner")
        .round(2)  # round to 2 digits
    )

    # outputs: scores
    scores = metrics.to_dict("records")[0]

    # outputs: percentile of scores
    percentiles = {
        "uniqueness_percentile": round(
            percentileofscore(metrics_dist_v3d["uniqueness"], scores["uniqueness"]), 2
        ),
        "readability_percentile": round(
            percentileofscore(metrics_dist_v3d["readability"], scores["readability"]), 2
        ),
        "mni_percentile": round(
            percentileofscore(metrics_dist_v3d["mni"], scores["mni"]), 2
        ),
    }

    # outputs: color of percentiles
    uniqueness_color = "yellow"  # the larger the better
    if percentiles["uniqueness_percentile"] < 40:
        uniqueness_color = "red"
    if percentiles["uniqueness_percentile"] >= 70:
        uniqueness_color = "green"

    readability_color = "yellow"  # the smaller the better
    if percentiles["readability_percentile"] < 40:
        readability_color = "green"
    if percentiles["readability_percentile"] >= 70:
        readability_color = "red"

    mni_color = "yellow"  # the larger the better
    if percentiles["mni_percentile"] < 40:
        mni_color = "red"
    if percentiles["mni_percentile"] >= 70:
        mni_color = "green"

    # comment the following line if you want to multiply uniqueness by 1000
    # (for better formatting in the output)
    scores["uniqueness"] = scores["uniqueness"] / 1000

    # final output
    colors = {
        "uniqueness_color": uniqueness_color,
        "readability_color": readability_color,
        "mni_color": mni_color,
    }

    return scores, percentiles, colors


application = app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        # get the image
        img = request.files["image"]

        # save the image
        img_path = wdir / "data/uploaded-images" / img.filename
        img.save(img_path)

        # identify objects
        obj_df = get_objects(img_path)

        # remove the image
        img_path.unlink()

        # calculate uniqueness
        uniqueness = get_uniqueness(obj_df, obj_freq)

        # calculate readability
        readability = get_readability(obj_df)

        # calculate mni
        mni = get_mni(obj_df, obj_mni)

        # combine all metrics
        scores, percentiles, colors = get_metrics(uniqueness, readability, mni)

        context = {
            # "title": "Article Writing Style Evaluation",
            "score": scores,
            "percent": percentiles,
            "color": colors,
        }
        # print(context)

        return render_template("output.html", context=context)

    return render_template("upload.html")


if __name__ == "__main__":
    app.run()
