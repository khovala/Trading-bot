import os
import pathlib
import subprocess
import shutil


def create_new_venv(
    project_name: str,
    requirements_file: str=None,
    force: bool=False
    ) -> None:
    venv_path = f'/opt/airflow/.venv_{project_name}'

    if pathlib.Path(venv_path).exists():
        if not force:
            return

    print(f'Creating venv at {venv_path}...')
    subprocess.run(['python3', '-m', 'venv', venv_path], check=True)

    if requirements_file:
        print(f'Installing dependencies from {requirements_file}...')
        subprocess.run([f'{venv_path}/bin/pip', 'install', '-r', requirements_file], check=True)

def delete_venv(project_name: str) -> None:
    venv_path = f'/opt/airflow/.venv_{project_name}'
    print(f'Cleaning up venv at {venv_path}...')
    shutil.rmtree(venv_path, ignore_errors=True)

if __name__ == '__main__':
    create_new_venv(project_name='crypto_sentiment', requirements_file='/opt/airflow/dags/scripts/requirements.txt')