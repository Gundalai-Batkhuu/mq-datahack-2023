"""
This is a boilerplate pipeline 'SVM'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["iris_raw",  "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="nb_split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "params:model_options"],
                outputs="NB",
                name="nb_train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["iris_raw", "SVC", "X_test", "y_test"],
                name="nb_evaluate_model_node",
                outputs="nb_metrics",
            )
        ]
    )

