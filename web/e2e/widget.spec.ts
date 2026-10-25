import http from "node:http";
import { expect, test } from "@playwright/test";
import { API, apiToken } from "./helpers";

let server: http.Server; let key = "";
test.beforeAll(async ({ request }) => {
  const h = await apiToken(request);
  key = (await (await request.get(`${API}/v1/widget`, { headers: h })).json()).widget_key;
  // a customer's page on an allowed origin (http://localhost:8080), embedding the widget with one script tag
  server = http.createServer((_, res) => {
    res.setHeader("content-type", "text/html");
    res.end(`<!doctype html><title>Shop</title><h1>Acme shop</h1><p>Some page content.</p>
      <script src="http://localhost:3000/widget.js" data-key="${key}" data-title="Ask Ava" data-color="#0f766e" async></script>`);
  }).listen(8080);
});
test.afterAll(() => server?.close());

test("widget: full voice call on a third-party site with live captions, mute, end and CSAT", async ({ page, request }) => {
  await page.goto("http://localhost:8080/");
  await page.getByRole("button", { name: /open voice assistant/i }).click();
  await expect(page.getByText("Start a voice conversation")).toBeVisible();
  await page.screenshot({ path: "../docs/screenshots/widget-start.png" });
  await page.getByRole("button", { name: "Start call" }).click();
  await expect(page.getByRole("status").filter({ hasText: /Listening|Speaking/ })).toBeVisible();           // connected
  // greeting is word-synced into the captions
  await expect(page.getByLabel("Live captions")).toContainText("Ava", { timeout: 30_000 });
  // the fake microphone asks about the return policy; the answer is grounded in the knowledge base
  await expect(page.getByLabel("Live captions")).toContainText(/thirty|30/i, { timeout: 60_000 });
  await expect(page.getByLabel("Live captions")).toContainText("return policy", { ignoreCase: true });   // user's own caption
  await page.screenshot({ path: "../docs/screenshots/widget-call.png" });
  await page.getByRole("button", { name: "Mute" }).click();
  await expect(page.getByRole("button", { name: "Unmute" })).toBeVisible();
  await page.getByRole("button", { name: "End call" }).click();
  await expect(page.getByText("Thanks for talking with us.")).toBeVisible();
  await page.getByRole("button", { name: "5 stars" }).click();
  await expect(page.getByText("Thank you for your feedback!")).toBeVisible();
  // CSAT reached the conversation record
  const h = await apiToken(request);
  await expect.poll(async () => (await (await request.get(`${API}/v1/conversations?limit=1`, { headers: h })).json())[0]?.csat, { timeout: 20_000 }).toBe(5);
});

test("widget: unlisted origin is refused with a clear message", async ({ page, request }) => {
  const h = await apiToken(request);
  await request.put(`${API}/v1/widget`, { headers: h, data: { allowed_origins: ["https://only-this-site.example"] } });
  try {
    await page.goto("http://localhost:8080/");
    await page.getByRole("button", { name: /open voice assistant/i }).click();
    await page.getByRole("button", { name: "Start call" }).click();
    await expect(page.getByRole("alert").filter({ hasText: "not allowed" })).toBeVisible();
  } finally {
    await request.put(`${API}/v1/widget`, { headers: h, data: { allowed_origins: ["http://localhost:3000", "http://localhost:8080"] } });
  }
});
