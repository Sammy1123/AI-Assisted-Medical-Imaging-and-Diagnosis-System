export default async function run(page, ui) {
  const out = {};

  // 1. Viewer page renders
  const snap0 = await ui.snapshot();
  out.viewerHasAnalyse = /Analyse/.test(snap0);
  out.disclaimerShown = /NOT FOR CLINICAL USE/.test(snap0);

  // 2. Upload the DICOM through the file input
  const input = page.locator('input[type="file"]');
  await input.setInputFiles(
    "D:\\AI-Assisted Medical Imaging and Diagnosis System\\data\\raw\\stage_2_test_images\\0000a175-0e68-4ca4-b1af-167204a7e0bc.dcm",
  );
  out.fileSelected = true;

  // 3. Click Analyse and wait for the result panel
  await page.getByRole("button", { name: "Analyse" }).click();
  await page
    .getByRole("heading", { name: "Result" })
    .waitFor({ timeout: 60000 });
  out.resultShown = true;

  // 4. Read the prediction
  out.label = await page.evaluate(() => {
    const ps = [...document.querySelectorAll("p")];
    const get = (k) =>
      ps.find((p) => p.textContent.startsWith(k))?.textContent ?? null;
    return {
      label: get("Label:"),
      prob: get("Probability:"),
      conf: get("Confidence:"),
    };
  });

  // 5. Grad-CAM image actually rendered with data
  out.gradcam = await page.evaluate(() => {
    const img = document.querySelector('img[alt="Grad-CAM"]');
    if (!img) return { present: false };
    return {
      present: true,
      loaded: img.complete && img.naturalWidth > 0,
      w: img.naturalWidth,
      h: img.naturalHeight,
    };
  });

  // 6. Radiologist review — fill ID + notes, approve
  const radRef = (await ui.snapshot()).match(/@(e\d+) textbox/)?.[1];
  const refs = (await ui.snapshot()).match(/@(e\d+)/g) || [];
  out.refsAfterResult = refs;

  // Fill the two textboxes (Radiologist ID, Notes)
  const boxes = page.locator(
    'input[placeholder="Radiologist ID"], input[placeholder="Notes (optional)"]',
  );
  await boxes.nth(0).fill("dr_browser_test");
  await boxes.nth(1).fill("Full-flow browser verification");
  await page.getByRole("button", { name: "✅ Approve" }).click();

  // 7. Wait for the confirmation
  await page
    .getByText("Decision recorded in audit log")
    .waitFor({ timeout: 15000 });
  out.reviewRecorded = true;

  // 8. Switch to Audit Log page
  await page.getByRole("button", { name: "Audit Log" }).click();
  await page.waitForTimeout(1500);
  out.auditPage = await page.evaluate(() =>
    document.body.innerText.slice(0, 600),
  );

  return out;
}
