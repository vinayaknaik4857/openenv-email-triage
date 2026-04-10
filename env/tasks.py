from __future__ import annotations

from env.models import EmailItem, SupportTask, TriageTarget


def get_tasks() -> list[SupportTask]:
    return [
        SupportTask(
            task_id="triage-easy-001",
            difficulty="easy",
            objective="Triage a straightforward duplicate-charge billing complaint.",
            instructions=(
                "Handle a straightforward billing problem. Determine category first, then priority/SLA, "
                "then draft a concise response with refund next steps."
            ),
            email=EmailItem(
                email_id="email-easy-001",
                sender="anna@example.com",
                subject="Charged twice for April invoice",
                body=(
                    "Hi support, I was billed twice this month and need a refund. "
                    "Please fix this today if possible."
                ),
                customer_tier="standard",
                created_at="2026-04-01T09:15:00Z",
            ),
            target=TriageTarget(
                category="billing",
                priority="high",
                response_sla_hours=8,
                required_response_keywords=["refund", "billing", "update"],
            ),
            success_notes=[
                "Customer is asking about a duplicate charge and wants a refund.",
                "A strong response should acknowledge the billing issue and promise an update.",
            ],
            max_steps=6,
        ),
        SupportTask(
            task_id="triage-medium-001",
            difficulty="medium",
            objective="Triage a production-impacting SSO incident for a premium customer team.",
            instructions=(
                "This is a service reliability issue affecting many users. Prioritize urgency and include "
                "investigation and ETA language in your response."
            ),
            email=EmailItem(
                email_id="email-medium-001",
                sender="it-admin@acme.io",
                subject="Intermittent SSO failures for multiple users",
                body=(
                    "Our team has recurring SSO login failures since yesterday afternoon. "
                    "Roughly 20 percent of employees cannot access dashboards. "
                    "Please investigate root cause and provide ETA."
                ),
                customer_tier="premium",
                created_at="2026-04-06T18:50:00Z",
            ),
            target=TriageTarget(
                category="technical",
                priority="urgent",
                response_sla_hours=2,
                required_response_keywords=["investigate", "eta", "incident"],
            ),
            success_notes=[
                "Many affected users and degraded access should drive urgent handling.",
                "The draft should mention investigation, incident handling, and an ETA commitment.",
            ],
            max_steps=6,
        ),
        SupportTask(
            task_id="triage-hard-001",
            difficulty="hard",
            objective="Triage a security-sensitive enterprise access governance issue before audit.",
            instructions=(
                "Treat this as a high-risk account governance issue. Classify under the best single category, "
                "set strict SLA, and draft response covering revocation and audit logging."
            ),
            email=EmailItem(
                email_id="email-hard-001",
                sender="cto@northstar-enterprise.com",
                subject="Security and access concerns after account ownership change",
                body=(
                    "We recently changed account ownership, and some former contractors still appear "
                    "to have API token access. We are preparing for an external compliance audit this week. "
                    "Need immediate guidance on revocation, logs, and account controls."
                ),
                customer_tier="enterprise",
                created_at="2026-04-07T11:05:00Z",
            ),
            target=TriageTarget(
                category="account",
                priority="urgent",
                response_sla_hours=1,
                required_response_keywords=["revoke", "audit", "access"],
            ),
            success_notes=[
                "This is an access-governance and security escalation for an enterprise account.",
                "The draft should cover revocation, audit logging, and immediate control review.",
            ],
            max_steps=7,
        ),
    ]
