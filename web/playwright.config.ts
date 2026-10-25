import path from "node:path";
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "e2e",
  timeout: 120_000,
  expect: { timeout: 45_000 },
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: process.env.CONSOLE_URL ?? "http://localhost:3000",
    permissions: ["microphone"],
    screenshot: "only-on-failure",
    launchOptions: {
      args: [
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        `--use-file-for-fake-audio-capture=${path.join(__dirname, "e2e/fixtures/mic.wav")}%noloop`,
        "--autoplay-policy=no-user-gesture-required",
      ],
    },
  },
});
