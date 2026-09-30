import { initializeApp } from "firebase/app";
import { connectAuthEmulator, getAuth } from "firebase/auth";

import { config } from "../config";

export const firebaseApp = initializeApp(config.firebase);
export const auth = getAuth(firebaseApp);

if (config.useAuthEmulator) {
  connectAuthEmulator(auth, "http://127.0.0.1:9099", { disableWarnings: true });
}
