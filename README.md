# robot_grasping_classifier

Dataset available: <https://www.kaggle.com/ugocupcic/grasping-dataset?select=shadow_robot_dataset.csv>
Download and extract archive from Kaggle
Put `shadow_robot_dataset.csv` in data/ folder (or anywhere and supply the correct path to `-i`)

To run with existing fine-tuned models: `python src/app.py -i data/shadow_robot_dataset.csv -m tuned_models.pkl`

To perform new grid search and run with new models `python src/app.py -i data/shadow_robot_dataset.csv -m <name>`
WARNING: Grid search may take a *very* long time.
