import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path

def initialize_data():
    try:
        import kaggle
    except ImportError("kagglehub is necessary to download the dataset") as e:
        raise e
    
    raw_path = Path("data/raw")

    if not raw_path.exists():
        raw_path.mkdir(parents=True, exist_ok=True)

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files("prashantsingh001/recipes-dataset-64k-dishes", path=raw_path, unzip=True)


def load_data(path: Path):
    df = pl.read_csv(path)
    return df


def parse_json_columns(df: pl.DataFrame):

    df = df.with_columns(
        pl.col("ingredients").str.json_decode(pl.List(pl.String)),
        pl.col("directions").str.json_decode(pl.List(pl.String)),
    )

    return df


if __name__ == '__main__':
    data_path = Path("data/raw/recipes.csv")

    df = load_data(path=data_path)

    df = parse_json_columns(df=df)
    
    print(df.head())

    print(df.select(
        pl.col("category").unique().len().alias("dish_categories"),
        pl.col("num_ingredients").max().alias("max_ingredients"),
        pl.col("num_ingredients").min().alias("min_ingredients"),
        pl.col("num_steps").max().alias("max_steps"),
        pl.col("num_steps").min().alias("min_steps"),
    ))

    fig, ax = plt.subplots(1, 2)

    ax[0].hist(df.select(pl.col("num_ingredients")), bins=35, width=0.8)
    ax[0].set_title("Number of ingredients")

    ax[1].hist(df.select(pl.col("num_steps")), bins=25, width=0.8)
    ax[1].set_title("Number of steps")

    plt.show()