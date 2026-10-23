import { expect, test } from "@playwright/test";
import path from "node:path";

const IMAGE = path.resolve(process.cwd(), "../datasets/demo/images/COCO_val2014_000000039769.jpg");
const DESCRIPTION = "Two cats are lying on a couch next to a remote control. A dog is sleeping on the couch. The cats are black.";

test("upload an image, verify claims, inspect evidence, see history", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Verify a description/ })).toBeVisible();
  await page.getByTestId("file-input").setInputFiles(IMAGE);
  await expect(page.getByAltText("Selected file preview")).toBeVisible();
  await page.getByLabel("Description to verify").fill(DESCRIPTION);
  await page.getByRole("button", { name: "Start verification" }).click();

  await expect(page).toHaveURL(/\/jobs\/[0-9a-f-]{36}$/);
  await expect(page.getByRole("heading", { name: "Summary" })).toBeVisible();

  const claims = page.getByRole("region", { name: "Claims" });
  const cats = claims.getByRole("button", { name: /Cats are visible\./ });
  await expect(cats).toContainText("Supported");
  const dog = claims.getByRole("button", { name: /A dog is visible\./ });
  await expect(dog).toContainText("Contradicted");

  // selecting a supported, localized claim draws evidence boxes over the original image
  await cats.click();
  await expect(page.getByRole("region", { name: "Claim detail" })).toContainText("Supporting evidence");
  await expect(page.getByAltText("Uploaded media")).toBeVisible();
  await expect(page.getByLabel("Evidence viewer").locator(".box-label").first()).toBeVisible();

  // a contradicted claim shows contradicting evidence and no fabricated region
  await dog.click();
  await expect(page.getByRole("region", { name: "Claim detail" })).toContainText(/Contradicting evidence \([1-9]/);
  await page.screenshot({ path: "../data/e2e/job-page.png", fullPage: true });

  await page.getByRole("link", { name: "History" }).first().click();
  await expect(page.getByRole("cell", { name: /Two cats are lying/ })).toBeVisible();

  await page.getByRole("link", { name: "Evaluation" }).click();
  await expect(page.getByRole("heading", { name: "Evaluation", exact: true })).toBeVisible();
  await expect(page.getByText(/No evaluation runs registered|Systems on the test split/)).toBeVisible();
});

test("unsupported file is rejected client-side and server errors are shown", async ({ page, request }) => {
  await page.goto("/");
  await page.getByTestId("file-input").setInputFiles({ name: "notes.pdf", mimeType: "application/pdf", buffer: Buffer.from("%PDF-1.4") });
  await expect(page.getByRole("alert")).toContainText("Unsupported format");
  const r = await request.post("/api/v1/verifications", {
    multipart: { file: { name: "x.png", mimeType: "image/png", buffer: Buffer.from("not really a png") }, description: "A dog." },
  });
  expect(r.status()).toBe(415);
  expect((await r.json()).error.code).toBe("unsupported_media_type");
});
