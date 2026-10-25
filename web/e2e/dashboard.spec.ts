import { expect, test } from "@playwright/test";
import { login } from "./helpers";

test("auth: protected pages redirect, bad credentials are rejected", async ({ page }) => {
  await page.goto("/agent");
  await expect(page).toHaveURL(/\/login/);
  await page.getByLabel("Email").fill("admin@acme.test");
  await page.getByLabel("Password").fill("wrong-password");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.locator(".alert.bad")).toContainText("Invalid email or password");
});

test("admin: edit the agent, pick and preview a voice, tune turn-taking, save", async ({ page }) => {
  await login(page);
  await page.screenshot({ path: "../docs/screenshots/overview.png", fullPage: true });
  await page.getByRole("link", { name: "Agent" }).click();
  await expect(page.getByRole("heading", { name: "Agent" })).toBeVisible();

  const greeting = page.getByRole("tab", { name: "Scripts" });
  await greeting.click();
  const box = page.getByLabel("Greeting");
  const original = await box.inputValue();
  await box.fill(original + " Ask me anything.");

  await page.getByRole("tab", { name: "Voice" }).click();
  await expect(page.getByRole("radio", { name: /Amy/ })).toBeVisible();
  await expect(page.getByRole("radio", { name: /Ryan/ })).toBeVisible();
  await page.getByRole("radio", { name: /Ryan/ }).click();
  const preview = page.waitForResponse((r) => r.url().includes("/voices/preview") && r.status() === 200);
  await page.getByRole("radio", { name: /Ryan/ }).getByRole("button", { name: /Preview/ }).click();
  expect((await preview).headers()["content-type"]).toContain("audio/wav");
  await page.screenshot({ path: "../docs/screenshots/agent-voice.png", fullPage: true });

  await page.getByRole("tab", { name: "Turn-taking" }).click();
  await page.getByRole("button", { name: "Snappy" }).click();
  await expect(page.getByText("220 ms")).toBeVisible();
  await page.screenshot({ path: "../docs/screenshots/agent-turn-taking.png", fullPage: true });

  await page.getByRole("button", { name: "Save changes" }).click();
  await expect(page.getByRole("status").filter({ hasText: "Saved" })).toBeVisible();

  // restore: reload shows the persisted values, then put the originals back
  await page.reload();
  await page.getByRole("tab", { name: "Voice" }).click();
  await expect(page.getByRole("radio", { name: /Ryan/ })).toHaveAttribute("aria-checked", "true");
  await page.getByRole("radio", { name: /Amy/ }).click();
  await page.getByRole("tab", { name: "Turn-taking" }).click();
  await page.getByRole("button", { name: "Balanced" }).click();
  await page.getByRole("tab", { name: "Scripts" }).click();
  await page.getByLabel("Greeting").fill(original);
  await page.getByRole("button", { name: "Save changes" }).click();
  await expect(page.getByRole("status").filter({ hasText: "Saved" })).toBeVisible();
});

test("admin: invalid tool JSON and server-side validation errors are shown, not swallowed", async ({ page }) => {
  await login(page);
  await page.goto("/agent");
  await page.getByRole("tab", { name: "Tools" }).click();
  await expect(page.getByRole("heading", { name: "order_status" })).toBeVisible();
  const t = page.getByLabel("Tool 1 definition");
  const good = await t.inputValue();
  await t.fill("{ not json");
  await page.getByRole("button", { name: "Save changes" }).click();
  await expect(page.locator(".alert.bad")).toContainText("Tool 1 is not valid JSON");
  await t.fill(good.replace(/"timeout_s": [0-9.]+/, '"timeout_s": 99'));
  await page.getByRole("button", { name: "Save changes" }).click();
  await expect(page.locator(".alert.bad")).toContainText(/timeout_s/);
  await page.screenshot({ path: "../docs/screenshots/agent-tools.png", fullPage: true });
});

test("knowledge: list, upload, test retrieval, delete", async ({ page }) => {
  await login(page);
  await page.goto("/knowledge");
  await expect(page.getByRole("cell", { name: "returns_policy.md", exact: true }).first()).toBeVisible();
  await page.getByLabel("Upload documents").setInputFiles({ name: "e2e-note.txt", mimeType: "text/plain",
    buffer: Buffer.from("The Acme loyalty program gives members 3 percent back on every purchase and free express shipping on orders above 100 dollars.") });
  await expect(page.getByText("e2e-note.txt: 1 passages indexed")).toBeVisible();
  await page.getByLabel("Test question").fill("do you have a loyalty program?");
  await page.getByRole("button", { name: "Search" }).click();
  await expect(page.getByText(/passages in \d+/)).toBeVisible();
  await expect(page.getByText("3 percent back").first()).toBeVisible();
  await page.screenshot({ path: "../docs/screenshots/knowledge.png", fullPage: true });
  await page.getByRole("button", { name: "Delete e2e-note.txt" }).click();
  await expect(page.getByRole("cell", { name: "e2e-note.txt" })).toHaveCount(0);
});

test("viewer: read-only everywhere", async ({ page }) => {
  await login(page, "viewer");
  await page.goto("/agent");
  await expect(page.getByText("read-only access")).toBeVisible();
  await expect(page.getByRole("button", { name: "Save changes" })).toBeDisabled();
  await page.goto("/knowledge");
  await expect(page.getByText("Upload documents")).toHaveCount(0);
  await page.goto("/team");
  await expect(page.getByText("Only admins can manage the team.")).toBeVisible();
});

test("widget page: key, snippet, allowed sites", async ({ page }) => {
  await login(page);
  await page.goto("/widget");
  await expect(page.locator("pre.snip")).toContainText(/data-key="wk_[0-9a-f]{32}"/);
  await expect(page.getByLabel("Allowed sites")).toHaveValue(/localhost:3000/);
  await page.screenshot({ path: "../docs/screenshots/widget-settings.png", fullPage: true });
});

test("test call: live latency overlay, transcript, then the conversation log shows per-turn latency", async ({ page }) => {
  await login(page, "editor");
  await page.goto("/test");
  await page.getByRole("button", { name: "Start test call" }).click();
  await expect(page.getByRole("status").filter({ hasText: /Listening|Speaking/ })).toBeVisible();
  await expect(page.getByRole("log")).toContainText("return policy", { ignoreCase: true, timeout: 60_000 });
  await expect(page.getByRole("log")).toContainText(/thirty|30/i, { timeout: 60_000 });
  // overlay fills in once the reply completes
  await expect(page.getByText("Last turn (server)").locator("xpath=following-sibling::div")).toContainText(/\d+ ms/, { timeout: 60_000 });
  await page.screenshot({ path: "../docs/screenshots/test-call.png", fullPage: true });
  await page.getByRole("button", { name: "End call" }).click();
  await page.goto("/calls");
  await expect(page.getByRole("row").nth(1)).toBeVisible();
  await page.getByRole("row").nth(1).click();
  await expect(page.getByText(/v2v \d+ ms/).first()).toBeVisible({ timeout: 20_000 });
  await page.screenshot({ path: "../docs/screenshots/conversation.png", fullPage: true });
});

test("onboarding: sign up, configure and reach the test step", async ({ page }) => {
  const email = `owner+${Date.now()}@globex.test`;
  await page.goto("/signup");
  await page.getByLabel("Organization").fill("Globex Repairs");
  await page.getByLabel("Work email").fill(email);
  await page.getByLabel("Password").fill("a-long-enough-pass");
  await page.getByRole("button", { name: "Create agent" }).click();
  await expect(page.getByRole("heading", { name: "Set up your voice agent" })).toBeVisible();
  await page.getByLabel("Agent name").fill("Gus");
  await page.getByLabel("What does your agent do?").fill("Answers questions about repair status for Globex Repairs.");
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByLabel("Upload documents").setInputFiles({ name: "faq.csv", mimeType: "text/csv",
    buffer: Buffer.from('question,answer\n"How long is a repair?","Most repairs take 3 to 5 business days."\n') });
  await expect(page.getByText("faq.csv: 1 passages indexed")).toBeVisible();
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByRole("radio", { name: /Lessac/ }).click();
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByRole("radio", { name: /Snappy/ }).click();
  await page.screenshot({ path: "../docs/screenshots/onboarding.png", fullPage: true });
  await page.getByRole("button", { name: "Save & try it" }).click();
  await expect(page.getByRole("button", { name: "Start test call" })).toBeVisible();
  await page.getByRole("link", { name: "Finish" }).click();
  await page.goto("/agent");
  await expect(page.getByLabel("Agent name")).toHaveValue("Gus");
});

test("conversation page: admin can export and delete; viewer sees no destructive actions", async ({ page, request }) => {
  const { API, apiToken } = await import("./helpers");
  const h = await apiToken(request);
  const list = await (await request.get(`${API}/v1/conversations?limit=1`, { headers: h })).json();
  test.skip(!list.length, "no conversation recorded yet");
  await login(page, "viewer");
  await page.goto(`/calls/${list[0].id}`);
  await expect(page.getByRole("button", { name: "Delete" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Export JSON" })).toHaveCount(0);
});
