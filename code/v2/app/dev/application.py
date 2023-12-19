import os
import pandas as pd

from flask import Flask, request, render_template
from pathlib import Path
from mmdet.apis import DetInferencer
from PIL import Image
from pyarrow.feather import read_feather

# set working directory
wdir = Path("/home/yu/OneDrive/Construal/code/v2/app/dev")
os.chdir(wdir)

# init global variables
#     score_threshold (float): threshold for the confidence score of the object
#     context (str): context of the image, either "kickstarter" or "general"
score_threshold = 0.1
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


def get_obj_freq(score_threshold):
    # load the object-level frequency table (Kick- and V3D-context)
    freq_kick = read_feather(
        f"data/freq_kick_p{int(score_threshold*100)}.feather",
        columns=["label", "freq"],
    )
    freq_v3d = read_feather(
        # for v3d, we only use one threahold (0.5)
        "data/freq_v3d_p50.feather",
        columns=["label", "freq"],
    )

    # Rename the 'freq' column to 'freq_kick'
    freq_v3d = freq_v3d.rename(columns={"freq": "freq_v3d"})
    freq_kick = freq_kick.rename(columns={"freq": "freq_kick"})

    # combine the two frequency tables into one, `freq`
    freq = pd.merge(freq_kick, freq_v3d, on="label", how="inner")

    return freq


obj_freq = get_obj_freq(score_threshold)


def get_obj_mni(score_threshold):
    """get the object-level MNI
    Returns:
        obj_mni: a DataFrame that contains the mni of each object

    """
    obj_mni_k10_kick = read_feather(
        f"data/obj_mni_kick_k10_p{int(score_threshold*100)}.feather"
    )
    obj_mni_k25_kick = read_feather(
        f"data/obj_mni_kick_k25_p{int(score_threshold*100)}.feather"
    )
    obj_mni_k50_kick = read_feather(
        f"data/obj_mni_kick_k50_p{int(score_threshold*100)}.feather"
    )
    obj_mni_k100_kick = read_feather(
        f"data/obj_mni_kick_k100_p{int(score_threshold*100)}.feather"
    )

    # load MNI based on V3D (for V3D we only use one threshold, 0.5)
    obj_mni_k10_v3d = read_feather("data/obj_mni_v3d_k10_p50.feather")
    obj_mni_k25_v3d = read_feather("data/obj_mni_v3d_k25_p50.feather")
    obj_mni_k50_v3d = read_feather("data/obj_mni_v3d_k50_p50.feather")
    obj_mni_k100_v3d = read_feather("data/obj_mni_v3d_k100_p50.feather")

    # rename columns in obj_mni_kick
    obj_mni_k10_kick = obj_mni_k10_kick.rename(columns={"mni": "mni_k10_kick"})
    obj_mni_k25_kick = obj_mni_k25_kick.rename(columns={"mni": "mni_k25_kick"})
    obj_mni_k50_kick = obj_mni_k50_kick.rename(columns={"mni": "mni_k50_kick"})
    obj_mni_k100_kick = obj_mni_k100_kick.rename(columns={"mni": "mni_k100_kick"})

    # rename columns in obj_mni_v3d
    obj_mni_k10_v3d = obj_mni_k10_v3d.rename(columns={"mni": "mni_k10_v3d"})
    obj_mni_k25_v3d = obj_mni_k25_v3d.rename(columns={"mni": "mni_k25_v3d"})
    obj_mni_k50_v3d = obj_mni_k50_v3d.rename(columns={"mni": "mni_k50_v3d"})
    obj_mni_k100_v3d = obj_mni_k100_v3d.rename(columns={"mni": "mni_k100_v3d"})

    # merge all MNI datasets
    obj_mni = (
        obj_mni_k10_kick.merge(obj_mni_k25_kick, on="obj", how="inner")
        .merge(obj_mni_k50_kick, on="obj", how="inner")
        .merge(obj_mni_k100_kick, on="obj", how="inner")
        .merge(obj_mni_k10_v3d, on="obj", how="inner")
        .merge(obj_mni_k25_v3d, on="obj", how="inner")
        .merge(obj_mni_k50_v3d, on="obj", how="inner")
        .merge(obj_mni_k100_v3d, on="obj", how="inner")
    )

    return obj_mni


obj_mni = get_obj_mni(score_threshold)


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
            - freq_kick (float): frequency of the object in the kickstarter context
            - freq_v3d (float): frequency of the object in the general context
    """

    # we only keep objects with score >= score_threshold
    obj_df = obj_df.loc[obj_df.score >= score_threshold]

    # compute the freq of each project by aggregating the freq of each object in the project
    uniqueness = (
        obj_df.merge(freq, on="label", how="inner")
        .groupby("name")
        .agg({"freq_kick": "sum", "freq_v3d": "sum"})
        .reset_index()
    )

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
        obj_df.loc[obj_df.score >= score_threshold]
        .groupby("name")
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
    obj_df = obj_df.loc[obj_df.score >= score_threshold]

    # compute project-level MNI
    out = (
        obj_df.merge(obj_mni, left_on="label", right_on="obj", how="inner")
        .groupby("name")
        .agg(
            {
                "mni_k10_kick": "sum",
                "mni_k25_kick": "sum",
                "mni_k50_kick": "sum",
                "mni_k100_kick": "sum",
                "mni_k10_v3d": "sum",
                "mni_k25_v3d": "sum",
                "mni_k50_v3d": "sum",
                "mni_k100_v3d": "sum",
            }
        )
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
    metrics = uniqueness.merge(readability, on="name", how="inner").merge(
        mni, on="name", how="inner"
    )

    # only select the columns we need
    if context == "kickstarter":
        out = metrics.loc[
            :, ["name", "freq_kick", "readability", "mni_k10_kick"]
        ].rename(columns={"freq_kick": "uniqueness", "mni_k10_kick": "mni"})

    elif context == "general":
        out = metrics.loc[:, ["name", "freq_v3d", "readability", "mni_k10_v3d"]].rename(
            columns={"freq_v3d": "uniqueness", "mni_k10_v3d": "mni"}
        )

    # only keep first 2 digits
    out = out.round(2)

    return out


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

        # calculate uniqueness
        uniqueness = get_uniqueness(obj_df, obj_freq)

        # calculate readability
        readability = get_readability(obj_df)

        # calculate mni
        mni = get_mni(obj_df, obj_mni)

        # combine all metrics
        scores = get_metrics(uniqueness, readability, mni).to_dict("records")[0]

        context = {
            "title": "Article Writing Style Evaluation",
            "score": scores,
        }
        print(context)

        return render_template("output.html", context=context)

    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)
