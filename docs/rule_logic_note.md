# Rule Logic Note

EnvLex-Lite uses five treaty-inspired environmental rule families. The rule families are treated as compliance templates that require domestic, permit-level, or administrative mapping before they can support a final violation conclusion.

## Rule families

| Rule family | Main logic |
|---|---|
| Basel-inspired | Hazardous-waste movement requires regulated waste, controlled movement, and valid consent/disposal documentation. |
| Rotterdam-inspired | Hazardous chemical export requires listed chemical status, export action, and import-country consent/export authorization. |
| Stockholm-inspired | Persistent organic pollutant use, storage, or disposal requires exemption and disposal/storage evidence. |
| Minamata-inspired | Mercury release or mercury-waste handling requires threshold/evidence-quality checks and handling records. |
| Montreal-inspired | Ozone-depleting substance import, use, or production requires license and quota checks. |

## Rule object fields

Each rule object may include:

- `rule_id`,
- `rule_family`,
- `regulated_item`,
- `controlled_action`,
- `required_document`,
- `exemption_condition`,
- `threshold_parameter`,
- `threshold_value`,
- `evidence_requirement`,
- `human_review_trigger`,
- `corrective_action_template`,
- `rule_description`.

## Validation logic

The rule-constrained validator checks:

1. rule applicability,
2. obligation existence,
3. factual breach,
4. evidence quality,
5. domestic mapping availability,
6. missing critical evidence,
7. structured-text conflict.

The validator is applied after ML prediction and is not trained as part of the neural model.
