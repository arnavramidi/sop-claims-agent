"""AWAIT_CONSENT: a listed representative waits for the policyholder's consent.

The status of the consent is data held by the database; this step only asks for it and says what came back. A
check is made when the state is entered from VERIFY_ID and afterwards only when the caller confidently asks for one.
Nothing about any claim is said, kept in a fact or sent to a model while the caller waits.
"""

from claims_agent.db.identity import check_consent
from claims_agent.engine import STEPS, StepOutcome, TurnContext
from claims_agent.form.read import confident_yes, picked
from claims_agent.reply.brief import Fact, ReplyPart
from claims_agent.states import State


# Consent facts never hold a claim-status word ("approved", "pending", ...): the reply check lets a status word through
# wherever a `consent` or `status` fact holds it, and the agent must never call a claim approved.
_FACT_TEXT = {
    "approved": "The policyholder has given consent.",
    "pending": "The policyholder's consent has not come through yet.",
    "denied": "The policyholder has refused consent.",
}
_FACT_OTHER = "The state of the policyholder's consent could not be read."


def consent_fact(status: str) -> Fact:
    return Fact("consent", _FACT_TEXT.get(status, _FACT_OTHER))


def _waiting(first: bool, status: str) -> ReplyPart:
    return ReplyPart(
        "consent_requested" if first else "consent_pending", (consent_fact(status),),
        "I've asked the policyholder for their consent. I can't discuss any claims until it comes through."
        if first else
        "The policyholder's consent hasn't come through yet. Please check back with me once they've replied.")


async def step(ctx: TurnContext) -> StepOutcome:
    t, limit = ctx.deps.thresholds, ctx.deps.settings.limits.consent_checks
    consent = ctx.memory.section("consent", State.AWAIT_CONSENT)
    fired = [{"rule": "await_consent"}]

    role = picked(ctx.form.caller_role, t) if ctx.form.caller_role is not None else None
    if role == "policyholder":  # a representative now says they are the policyholder: a person must look at it
        return StepOutcome(State.AWAIT_CONSENT, handover=("rule", "role_switch"), fired=fired)

    recheck = ctx.form.consent_recheck is not None and confident_yes(ctx.form.consent_recheck, t)
    asked = ctx.entered_by_hop or recheck
    if not asked:
        status = consent.last_status or "pending"
        return StepOutcome(State.AWAIT_CONSENT, [_waiting(False, status)], fired=fired)

    first = ctx.entered_by_hop and not consent.requested
    consent.requested = True
    status = await check_consent(ctx.conn, ctx.conversation.id)
    consent.checks_used += 1
    consent.last_status = status
    fired.append({"rule": "check_consent", "status": status, "checks_used": consent.checks_used})
    ctx.record("consent_check", number=consent.checks_used, script=ctx.conversation.consent_script, result=status)

    if status == "approved":
        # The database binds the customer itself; the reply may say "verified" only if it really did.
        row = await (await ctx.conn.execute(
            "SELECT verified_party_id FROM conversations WHERE id = %s", [ctx.conversation.id])).fetchone()
        if row is None or row[0] is None:
            return StepOutcome(State.AWAIT_CONSENT, handover=("rule", "consent_unknown"), fired=fired)
        return StepOutcome(State.RESOLVE_INTENT, [ReplyPart(
            "verified", (consent_fact(status),), "Thank you, the policyholder has given consent and you're verified.")], hop=True, fired=fired)
    if status == "pending":
        if consent.checks_used >= limit:
            return StepOutcome(State.AWAIT_CONSENT, handover=("rule", "consent_timeout"), fired=fired)
        return StepOutcome(State.AWAIT_CONSENT, [_waiting(first, status)], fired=fired)
    if status == "denied":
        return StepOutcome(State.CLOSED, [ReplyPart(
            "consent_denied", (consent_fact(status),),
            "The policyholder hasn't given consent, so I can't discuss their claims.")], fired=fired)
    return StepOutcome(State.AWAIT_CONSENT, handover=("rule", "consent_unknown"), fired=fired)


STEPS[State.AWAIT_CONSENT] = step
