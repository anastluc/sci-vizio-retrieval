import { type ReactNode, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../api/client";
import type { Plan } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { Logo } from "../components/Logo";
import { formatPrice } from "../lib/format";

// Shown if the API can't be reached; the API's /plans is the source of truth.
const FALLBACK_PLANS: Plan[] = [
  { id: "free", name: "Free", price_monthly_usd: 0, max_documents: 10, max_file_mb: 20, max_figures_per_document: 40 },
  { id: "premium", name: "Premium", price_monthly_usd: 9, max_documents: 500, max_file_mb: 100, max_figures_per_document: 200 },
];

export function PlanCard({ plan, action }: { plan: Plan; action?: ReactNode }) {
  return (
    <div className={`plan-card ${plan.id === "premium" ? "plan-card-featured" : ""}`}>
      <h3>{plan.name}</h3>
      <p className="plan-price">
        {formatPrice(plan.price_monthly_usd)}
        {plan.price_monthly_usd > 0 && <span className="muted"> / month</span>}
      </p>
      <ul className="plan-features">
        <li>Up to <strong>{plan.max_documents}</strong> PDFs</li>
        <li>Files up to {plan.max_file_mb} MB</li>
        <li>Up to {plan.max_figures_per_document} figures analysed per PDF</li>
        <li>Collections, search &amp; filters</li>
      </ul>
      {action}
    </div>
  );
}

/** A static preview of a search: the query and three matching figure plates. */
function HeroDemo() {
  const ink = "#18222f";
  const blue = "#1f5fbf";
  const orange = "#d9822b";
  return (
    <div className="hero-demo" aria-label="Example search">
      <div className="hero-query">Search: <strong>line chart of validation loss during training</strong></div>
      <div className="hero-figures">
        <figure>
          <svg viewBox="0 0 160 110" role="img" aria-label="Line chart">
            <path d="M20 10v80h130" fill="none" stroke={ink} strokeWidth="1" />
            <path d="M22 22 C45 60 65 72 90 78 S130 84 148 85" fill="none" stroke={blue} strokeWidth="2" />
            <path d="M22 30 C45 58 65 66 90 70 S130 70 148 74" fill="none" stroke={orange} strokeWidth="2" strokeDasharray="4 3" />
          </svg>
          <figcaption><b>Validation loss by epoch</b>, page 7 of Scaling laws for sparse transformers</figcaption>
        </figure>
        <figure>
          <svg viewBox="0 0 160 110" role="img" aria-label="Line chart">
            <path d="M20 10v80h130" fill="none" stroke={ink} strokeWidth="1" />
            <polyline points="22,80 40,62 58,55 76,40 94,36 112,28 130,25 148,22" fill="none" stroke={blue} strokeWidth="2" />
            <polyline points="22,84 40,74 58,70 76,60 94,58 112,54 130,52 148,50" fill="none" stroke={orange} strokeWidth="2" />
          </svg>
          <figcaption><b>Accuracy over training steps</b>, page 4 of Curriculum learning for small models</figcaption>
        </figure>
        <figure>
          <svg viewBox="0 0 160 110" role="img" aria-label="Bar chart">
            <path d="M20 10v80h130" fill="none" stroke={ink} strokeWidth="1" />
            {[62, 48, 30, 40, 22].map((h, i) => (
              <rect key={i} x={30 + i * 24} y={90 - h} width="14" height={h} fill={i === 2 ? orange : blue} />
            ))}
          </svg>
          <figcaption><b>Final loss per optimiser</b>, page 9 of A benchmark of adaptive optimisers</figcaption>
        </figure>
      </div>
    </div>
  );
}

export function Landing() {
  const { user } = useAuth();
  const [plans, setPlans] = useState<Plan[]>(FALLBACK_PLANS);

  useEffect(() => {
    api.plans().then(setPlans).catch(() => undefined);
  }, []);

  const cta = user ? { to: "/app/library", label: "Open your library" } : { to: "/login?mode=signup", label: "Start free" };

  return (
    <div className="landing">
      <header className="topbar topbar-public">
        <Link to="/" className="brand"><Logo /> Sci-Vizio</Link>
        <div className="topbar-right">
          {user ? <Link className="btn btn-primary btn-sm" to="/app/library">Open app</Link>
            : <Link className="btn btn-ghost btn-sm" to="/login">Sign in</Link>}
        </div>
      </header>

      <section className="hero">
        <div className="hero-inner">
          <h1>Find the figure you remember, not the paper it was in.</h1>
          <p className="lead">
            Upload your research PDFs. Sci-Vizio pulls out every chart, diagram and table, describes each one
            with a vision model, and lets you search them in plain language.
          </p>
          <div><Link className="btn btn-primary btn-lg" to={cta.to}>{cta.label}</Link></div>
          <HeroDemo />
        </div>
      </section>

      <section className="steps">
        <div><span className="step-num">1</span><h3>Upload</h3><p>Drop in PDFs and organise them into collections.</p></div>
        <div><span className="step-num">2</span><h3>Understand</h3><p>Each figure gets a structured description: type, axes, labels, key patterns.</p></div>
        <div><span className="step-num">3</span><h3>Search</h3><p>Ask for “bar chart comparing model accuracy” and filter by type, collection or paper.</p></div>
      </section>

      <section className="pricing" id="pricing">
        <h2>Pricing</h2>
        <div className="plan-grid">
          {plans.map((p) => (
            <PlanCard key={p.id} plan={p}
              action={<Link className={`btn ${p.id === "premium" ? "btn-primary" : "btn-secondary"}`} to={cta.to}>
                {p.id === "premium" ? "Start free, upgrade anytime" : cta.label}
              </Link>} />
          ))}
        </div>
      </section>

      <footer className="footer muted small">© {new Date().getFullYear()} Sci-Vizio</footer>
    </div>
  );
}
