const env = import.meta.env;

export const config = {
  apiBaseUrl: (env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, ""),
  firebase: {
    apiKey: env.VITE_FIREBASE_API_KEY || "demo-api-key",
    authDomain: env.VITE_FIREBASE_AUTH_DOMAIN,
    projectId: env.VITE_FIREBASE_PROJECT_ID || "demo-sci-vizio",
    appId: env.VITE_FIREBASE_APP_ID,
  },
  useAuthEmulator: env.VITE_USE_AUTH_EMULATOR === "true",
};
