import { type FormEvent, useState } from "react";
import { Link, Navigate, useLocation, useSearchParams } from "react-router-dom";

import { authErrorMessage, useAuth } from "../auth/AuthContext";
import { Logo } from "../components/Logo";
import { Banner } from "../components/ui";

type Mode = "signin" | "signup" | "reset";

export function Login() {
  const { user, signIn, signUp, signInWithGoogle, resetPassword } = useAuth();
  const [params] = useSearchParams();
  const location = useLocation();
  const [mode, setMode] = useState<Mode>(params.get("mode") === "signup" ? "signup" : "signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  const from = (location.state as { from?: string } | null)?.from ?? "/app/library";
  if (user) return <Navigate to={from} replace />;

  async function run(action: () => Promise<void>) {
    setBusy(true);
    setError(null);
    setInfo(null);
    try {
      await action();
    } catch (e) {
      setError(authErrorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (mode === "signin") void run(() => signIn(email, password));
    else if (mode === "signup") void run(() => signUp(email, password, name.trim()));
    else void run(async () => {
      await resetPassword(email);
      setInfo("If an account exists for that email, a reset link is on its way.");
    });
  }

  const titles: Record<Mode, string> = {
    signin: "Sign in",
    signup: "Create your free account",
    reset: "Reset your password",
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <Link to="/" className="brand brand-center"><Logo size={28} /> Sci-Vizio</Link>
        <h1>{titles[mode]}</h1>

        {error && <Banner tone="error">{error}</Banner>}
        {info && <Banner tone="success">{info}</Banner>}

        {mode !== "reset" && (
          <>
            <button className="btn btn-secondary btn-block" disabled={busy} onClick={() => void run(signInWithGoogle)}>
              Continue with Google
            </button>
            <div className="divider"><span>or</span></div>
          </>
        )}

        <form onSubmit={onSubmit} className="form">
          {mode === "signup" && (
            <label>Name
              <input value={name} onChange={(e) => setName(e.target.value)} autoComplete="name" maxLength={200} />
            </label>
          )}
          <label>Email
            <input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" />
          </label>
          {mode !== "reset" && (
            <label>Password
              <input type="password" required minLength={6} value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete={mode === "signup" ? "new-password" : "current-password"} />
            </label>
          )}
          <button type="submit" className="btn btn-primary btn-block" disabled={busy}>
            {busy ? "Please wait…" : mode === "signin" ? "Sign in" : mode === "signup" ? "Create account" : "Send reset link"}
          </button>
        </form>

        <p className="auth-switch">
          {mode === "signin" && (<>
            <span>New here? <button className="link" onClick={() => setMode("signup")}>Create an account</button></span>
            <span><button className="link" onClick={() => setMode("reset")}>Forgot your password?</button></span>
          </>)}
          {mode === "signup" && (<>Already have an account? <button className="link" onClick={() => setMode("signin")}>Sign in</button></>)}
          {mode === "reset" && (<button className="link" onClick={() => setMode("signin")}>Back to sign in</button>)}
        </p>
      </div>
    </div>
  );
}
