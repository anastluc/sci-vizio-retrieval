import { Link, NavLink, Navigate, Outlet, useLocation } from "react-router-dom";

import { useAuth } from "../auth/AuthContext";
import { MeProvider, useMe } from "../auth/MeContext";
import { Logo } from "./Logo";

export function RequireAuth() {
  const { user, loading } = useAuth();
  const location = useLocation();
  if (loading) return <div className="page-center"><span className="spinner" aria-label="Loading" /></div>;
  if (!user) return <Navigate to="/login" replace state={{ from: location.pathname + location.search }} />;
  return (
    <MeProvider>
      <AppShell />
    </MeProvider>
  );
}

function AppShell() {
  const { me, error } = useMe();
  const { signOut } = useAuth();

  return (
    <div className="app">
      <header className="topbar">
        <Link to="/app/library" className="brand"><Logo /> Sci-Vizio</Link>
        <nav className="topnav" aria-label="Main">
          <NavLink to="/app/library">Library</NavLink>
          <NavLink to="/app/search">Search</NavLink>
          <NavLink to="/app/account">Account</NavLink>
        </nav>
        <div className="topbar-right">
          {me && (
            <Link to="/app/account" className={`plan-pill plan-${me.plan.id}`}>
              {me.plan.name}
            </Link>
          )}
          <button className="btn btn-ghost btn-sm" onClick={() => void signOut()}>Sign out</button>
        </div>
      </header>
      {error && <div className="banner banner-error" role="alert">{error}</div>}
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}
