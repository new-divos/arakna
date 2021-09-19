#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import current_app
from flask.cli import with_appcontext
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from arakna.process.h2qs.stats import Stats
from arakna.process.h2qs.transform import transform


@transform.command("isotonic")  # type: ignore
@with_appcontext
def isotonic() -> None:
    file_path = current_app.config["DATASET_PATH"]
    try:
        df = pd.read_parquet(file_path, engine="auto")
        if df.empty:
            raise ValueError("The raw dataset is empty")

        current_app.logger.info(
            f"The {df.shape[0]} rows were read from the file {file_path}"
        )

        h_train, h_test, _, ps_test, qs_train, qs_test = train_test_split(
            df.hurst,
            df.population_size,
            df.enqueued_max,
            test_size=current_app.config["TEST_RATE"],
            random_state=current_app.config["RANDOM_SEED"],
        )
        current_app.logger.info(
            f"The train sample contains {h_train.shape[0]} rows. "
            f"The test sample contains {h_test.shape[0]} rows"
        )

        model = IsotonicRegression(y_min=0.0, increasing="auto", out_of_bounds="clip")
        model.fit(h_train, qs_train)
        current_app.logger.info("The isotonic regression model was built")

        out_path = current_app.config["PROCESSED_ROOT"] / isotonic.name
        out_path.mkdir(parents=True, exist_ok=True)
        file_key = current_app.config["DATASET_PATH"].stem
        out_file_path = out_path / f"{file_key}.pkl"
        joblib.dump(model, out_file_path)
        current_app.logger.info(
            f"The isotonic regression model was saved to {out_file_path}"
        )

        qs_train_pred = model.predict(h_train)
        qs_test_pred = model.predict(h_test)

        stats = Stats(
            h_test.values,
            qs_test.values,
            qs_test_pred,
            ps_test,
            current_app.config["BETA_PLUS"],
            current_app.config["BETA_MINUS"],
            current_app.config["DELTA"],
        )

        out_file_path = out_path / f"{file_key}.test.parquet"
        stats.to_parquet(out_file_path)
        current_app.logger.info(
            f"The predictions were saved to the file {out_file_path}"
        )

        meta = dict(
            dataset=current_app.config["DATASET"],
            file=current_app.config["DATASET_PATH"].name,
            test_rate=current_app.config["TEST_RATE"],
            random_seed=current_app.config["RANDOM_SEED"],
            beta_plus=current_app.config["BETA_PLUS"],
            beta_minus=current_app.config["BETA_MINUS"],
            delta=current_app.config["DELTA"],
            timestamp=f"{datetime.now():%Y-%m-%d %H:%M:%S}",
            model="isotonic",
        )

        metrics = dict(
            train=Stats.metrics(qs_train, qs_train_pred),
            test=Stats.metrics(qs_test, qs_test_pred),
            performance={prop.name: prop.to_dict() for prop in stats},
        )

        h_points = np.linspace(0.5, 1.0, current_app.config["POINTS_NUMBER"])
        qs_points = np.ceil(model.predict(h_points)).astype(int)

        out_file_path = out_path / f"{file_key}.json"
        with open(out_file_path, "w", encoding="utf-8") as fout:
            json.dump(
                dict(
                    meta=meta,
                    metrics=metrics,
                    prediction=dict(
                        hurst=h_points.tolist(), queue_size=qs_points.tolist()
                    ),
                ),
                fout,
                ensure_ascii=False,
                indent=4,
            )
        current_app.logger.info(
            f"The transformation totals were saved to the file {out_file_path}"
        )

    except Exception as e:
        current_app.logger.error(
            f"The exception {e} occurs during transforming the file {file_path}"
        )

    finally:
        current_app.logger.info(
            "The transformation using the isotonic regression has been finished"
        )
