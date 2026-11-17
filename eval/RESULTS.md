# Résultats de l'évaluation Ragas

- Date : 2026-11-17
- Modèle RAG : `qwen2.5:3b`, juge : `qwen2.5:7b`, 12 questions

## Moyennes

| Métrique | Score |
|---|---|
| faithfulness | 0.89 |
| answer_relevancy | 0.51 |
| llm_context_precision_with_reference | 0.92 |
| context_recall | 0.92 |

## Détail par question

| Question | faithfulness | answer_relevancy | llm_context_precision_with_reference | context_recall |
|---|---|---|---|---|
| Combien de jours de télétravail par semaine un salarié peut-il prendre ? | 1.00 | 0.46 | 1.00 | 1.00 |
| Quel jour la présence au bureau est-elle obligatoire ? | 1.00 | 0.79 | 1.00 | 1.00 |
| Quel est le montant de l'indemnité de télétravail ? | 1.00 | 0.59 | 1.00 | 1.00 |
| Combien de jours de congés et de RTT ont les salariés ? | 0.67 | 0.48 | 1.00 | 1.00 |
| Quel est le plafond de remboursement pour une nuit d'hôtel à Lyon ? | 1.00 | 0.69 | 1.00 | 1.00 |
| Quand a lieu la journée d'intégration des nouveaux arrivants ? | 1.00 | 0.58 | 1.00 | 1.00 |
| Quelle prime reçoit un technicien pour une semaine d'astreinte ? | 1.00 | 0.46 | 1.00 | 1.00 |
| Quel est le délai d'intervention sur site pour un incident P1 ? | 0.00 | 0.30 | 1.00 | 1.00 |
| À partir de quelle perte de production un incident est-il classé P2 ? | 1.00 | 0.64 | 1.00 | 1.00 |
| Qui faut-il prévenir si un incident P1 n'est pas résolu au bout de 2 heures ? | 1.00 | 0.67 | 1.00 | 1.00 |
| Dans quel outil sont publiés le planning d'astreinte et les demandes de télétravail ? | 1.00 | 0.50 | 1.00 | 1.00 |
| Quel est le chiffre d'affaires annuel d'Hélioria Énergies ? | 1.00 | 0.00 | 0.00 | 0.00 |
