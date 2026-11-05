export default async function run(page, ui) {
  const out = {};
  // go to Segmentation page
  await page.getByRole("button", { name: "Segmentation" }).click();
  await page.getByText("Spleen Segmentation").waitFor({ timeout: 10000 });
  out.pageLoaded = true;

  // upload a real spleen volume
  const vol =
    "D:\\AI-Assisted Medical Imaging and Diagnosis System\\data\\raw\\Task09_Spleen\\imagesTr\\spleen_2.nii.gz";
  await page.locator('input[type="file"]').setInputFiles(vol);
  await page.getByRole("button", { name: "Segment", exact: true }).click();

  // wait for result panel
  await page.getByText("Masked axial slices").waitFor({ timeout: 180000 });
  out.result = await page.evaluate(() => {
    const ps = [...document.querySelectorAll("p")].map((p) => p.textContent);
    return ps.filter((t) => /Masked|Segmented/.test(t));
  });

  // take a screenshot of the overlay
  await page.screenshot({
    path: "D:\\AI-Assisted Medical Imaging and Diagnosis System\\evaluation\\seg_browser.png",
  });
  out.overlayVisible = await page.evaluate(() => {
    const img = document.querySelector('img[alt="Segmentation overlay"]');
    return img
      ? { loaded: img.complete && img.naturalWidth > 0, w: img.naturalWidth }
      : false;
  });

  // audit log should now include the segment event
  await page.getByRole("button", { name: "Audit Log" }).click();
  await page.waitForTimeout(1200);
  out.auditHasSegment = (
    await page.evaluate(() => document.body.innerText)
  ).includes("segment");
  return out;
}
