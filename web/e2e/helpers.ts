import { APIRequestContext, expect, Page } from "@playwright/test";

export const API = process.env.API_URL ?? "http://localhost:8000";
export const PASSWORD = process.env.ACME_ADMIN_PASSWORD ?? "acme-admin-pass-123";

export async function apiToken(req: APIRequestContext, role = "admin") {
  const r = await req.post(`${API}/v1/auth/login`, { data: { email: `${role}@acme.test`, password: PASSWORD } });
  expect(r.ok()).toBeTruthy();
  return { Authorization: `Bearer ${(await r.json()).token}` };
}

export async function login(page: Page, role = "admin") {
  await page.goto("/login");
  await page.getByLabel("Email").fill(`${role}@acme.test`);
  await page.getByLabel("Password").fill(PASSWORD);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("heading", { name: "Overview" })).toBeVisible();
}
