import json
import pathlib

import boto3
import typer

import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from src.settings import LOGGER


app = typer.Typer()

@app.command()
def push_to_s3(
    model_path: pathlib.Path,
    bucket_name: str,
    reports_path: pathlib.Path
    ) -> None:
    with open(f'{reports_path}/best_model.json', 'r') as f:
        best_model = json.load(f)
    if best_model['best_model'] == 'old':
        LOGGER.info('Skipping pushing stage...')
        return

    s3_session = boto3.session.Session()
    s3_client = s3_session.client(
        service_name='s3',
        endpoint_url='https://storage.yandexcloud.net'
    )
    s3_client.upload_file(
        model_path,
        bucket_name,
        'crypto_sentimnet_clf_new/crypto_sentiment_clf.joblib'
        )

if __name__ == '__main__':
    app()