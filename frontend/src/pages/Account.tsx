import { type FormEvent, useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

import { api } from "../api/client";
import type { Plan } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { useMe } from "../auth/MeContext";
import { Banner, UsageMeter } from "../components/ui";
import { formatBytes, formatDate, formatPrice } from "../lib/format";
import { PlanCard } from "./Landing";

export function Account() {
  const { me, refresh } = useMe();
  const { user, resetPassword, deleteFirebaseUser } = useAuth();
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const [plans, setPlans] = useState<Plan[]>([]);
  const [name, setName] = useState("");
  const [message, setMessage] = useState<{ tone: "success" | "error" | "info"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [confirmText, setConfirmText] = useState("");

  useEffect(() => { api.plans().then(setPlans).catch(() => undefined); }, []);
  // Usage changes as documents finish processing, so reload it whenever the page opens.
  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => { if (me) setName(me.display_name ?? ""); }, [me]);

  // Returning from Stripe Checkout: the webhook may land a moment later, so poll briefly.
  const checkout = params.get("checkout");
  useEffect(() => {
    if (checkout === "cancelled") {
      setMessage({ tone: "info", text: "Checkout cancelled — you're still on the free plan." });
      setParams({}, { replace: true });
      return;
    }
    if (checkout !== "success") return;
    setMessage({ tone: "info", text: "Payment received — activating Premium…" });
    let attempts = 0;
    const timer = setInterval(async () => {
      attempts += 1;
      const latest = await refresh();
      if (latest?.plan.id === "premium" || attempts >= 10) {
        clearInterval(timer);
        setParams({}, { replace: true });
        setMessage(latest?.plan.id === "premium"
          ? { tone: "success", text: "Welcome to Premium! Your new limits are active." }
          : { tone: "info", text: "Your upgrade is still being confirmed. Refresh in a minute." });
      }
    }, 2000);
    return () => clearInterval(timer);
  }, [checkout, refresh, setParams]);

  async function run(action: () => Promise<void>) {
    setBusy(true);
    setMessage(null);
    try {
      await action();
    } catch (e) {
      setMessage({ tone: "error", text: e instanceof Error ? e.message : "Something went wrong" });
    } finally {
      setBusy(false);
    }
  }

  function saveProfile(e: FormEvent) {
    e.preventDefault();
    void run(async () => {
      await api.updateMe(name);
      await refresh();
      setMessage({ tone: "success", text: "Profile saved." });
    });
  }

  const goTo = (fetchUrl: () => Promise<{ url: string }>) =>
    run(async () => { window.location.assign((await fetchUrl()).url); });

  function deleteAccount() {
    void run(async () => {
      await api.deleteMe();
      await deleteFirebaseUser();
      navigate("/", { replace: true });
    });
  }

  if (!me) return <div className="page-center"><span className="spinner" aria-label="Loading" /></div>;

  const premium = plans.find((p) => p.id === "premium");
  const usesPassword = user?.providerData.some((p) => p.providerId === "password");

  return (
    <div className="content narrow">
      <h1>Account</h1>
      {message && <Banner tone={message.tone} onClose={() => setMessage(null)}>{message.text}</Banner>}

      <section className="card">
        <h2>Plan &amp; usage</h2>
        <p>
          You're on the <strong>{me.plan.name}</strong> plan
          {me.plan.price_monthly_usd > 0 && <> ({formatPrice(me.plan.price_monthly_usd)}/month)</>}.
          {me.subscription_status === "past_due" && <span className="text-error"> Your last payment failed — please update your card.</span>}
          {me.plan.id === "premium" && me.current_period_end && (
            <span className="muted"> Current period ends {formatDate(me.current_period_end)}.</span>
          )}
        </p>
        <UsageMeter label="PDFs" used={me.usage.documents} limit={me.plan.max_documents} />
        <p className="muted small">
          {me.usage.figures} figures indexed from {formatBytes(me.usage.storage_bytes)} of PDFs.
          Your plan accepts files up to {me.plan.max_file_mb} MB and analyses up to {me.plan.max_figures_per_document} figures per PDF.
        </p>

        {!me.billing_enabled ? (
          <p className="muted small">Billing isn't enabled on this deployment.</p>
        ) : me.plan.id === "free" ? (
          premium && (
            <div className="upgrade">
              <PlanCard plan={premium} action={
                <button className="btn btn-primary" disabled={busy} onClick={() => void goTo(api.checkout)}>
                  Upgrade for {formatPrice(premium.price_monthly_usd)}/month
                </button>} />
            </div>
          )
        ) : (
          <button className="btn btn-secondary" disabled={busy} onClick={() => void goTo(api.billingPortal)}>
            Manage billing &amp; invoices
          </button>
        )}
      </section>

      <section className="card">
        <h2>Profile</h2>
        <form className="form" onSubmit={saveProfile}>
          <label>Email<input value={me.email ?? ""} disabled /></label>
          <label>Display name
            <input value={name} maxLength={200} onChange={(e) => setName(e.target.value)} />
          </label>
          <div><button className="btn btn-primary btn-sm" disabled={busy}>Save</button></div>
        </form>
      </section>

      {usesPassword && me.email && (
        <section className="card">
          <h2>Password</h2>
          <p className="muted">We'll email you a link to set a new password.</p>
          <button className="btn btn-secondary btn-sm" disabled={busy}
            onClick={() => void run(async () => {
              await resetPassword(me.email!);
              setMessage({ tone: "success", text: `Password reset email sent to ${me.email}.` });
            })}>Send reset email</button>
        </section>
      )}

      <section className="card card-danger">
        <h2>Delete account</h2>
        <p>
          Permanently deletes your PDFs, figures and collections
          {me.plan.id === "premium" && " and cancels your subscription"}. This can't be undone.
        </p>
        <div className="form">
          <label>
            <span>Type <strong>{me.email ?? "DELETE"}</strong> to confirm</span>
            <input value={confirmText} onChange={(e) => setConfirmText(e.target.value)} />
          </label>
        </div>
        <button className="btn btn-danger btn-sm" disabled={busy || confirmText !== (me.email ?? "DELETE")}
          onClick={deleteAccount}>Delete my account</button>
      </section>
    </div>
  );
}
