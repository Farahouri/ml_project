"""
deploiement_prefect.py
Déploie les flows Prefect avec une planification quotidienne.
"""

from prefect import serve

from pipeline_prefect import all_flow, train_flow, evaluate_flow, code_flow

if __name__ == "__main__":
    all_deploy = all_flow.to_deployment(
        name="ml-pipeline-all",
        cron="0 8 * * *",  # tous les jours à 08:00
    )
    train_deploy = train_flow.to_deployment(
        name="ml-pipeline-train",
        cron="0 2 * * *",  # tous les jours à 02:00
    )
    evaluate_deploy = evaluate_flow.to_deployment(name="ml-pipeline-evaluate")
    code_deploy = code_flow.to_deployment(name="ml-pipeline-code")

    serve(all_deploy, train_deploy, evaluate_deploy, code_deploy)
