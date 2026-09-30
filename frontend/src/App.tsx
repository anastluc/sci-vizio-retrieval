import { BrowserRouter, Link, Navigate, Route, Routes } from "react-router-dom";

import { AuthProvider } from "./auth/AuthContext";
import { RequireAuth } from "./components/AppLayout";
import { Account } from "./pages/Account";
import { Landing } from "./pages/Landing";
import { Library } from "./pages/Library";
import { Login } from "./pages/Login";
import { Search } from "./pages/Search";

function NotFound() {
  return (
    <div className="page-center">
      <h1>Page not found</h1>
      <Link to="/">Go home</Link>
    </div>
  );
}

export function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/login" element={<Login />} />
          <Route path="/app" element={<RequireAuth />}>
            <Route index element={<Navigate to="library" replace />} />
            <Route path="library" element={<Library />} />
            <Route path="search" element={<Search />} />
            <Route path="account" element={<Account />} />
          </Route>
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
