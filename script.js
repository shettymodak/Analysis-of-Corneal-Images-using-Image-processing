const PREDICTION_API = {
  endpoint: "/predict",
  fileField: "image",
  classes: ["control", "stress", "recovery"],
};  

const form = document.querySelector("#predict-form");
const fileInput = document.querySelector("#image-input");
const uploadArea = document.querySelector("#upload-area");
const fileName = document.querySelector("#file-name");
const uploadError = document.querySelector("#upload-error");
const apiError = document.querySelector("#api-error");
const preview = document.querySelector("#image-preview");
const previewEmpty = document.querySelector("#preview-empty");
const predictButton = document.querySelector("#predict-button");
const buttonLabel = predictButton.querySelector(".button-label");
const resultSection = document.querySelector("#result-section");
const predictedClass = document.querySelector("#predicted-class");
const confidenceValue = document.querySelector("#confidence-value");
const confidenceBar = document.querySelector("#confidence-bar");
const segmentationOverlay = document.querySelector("#segmentation-overlay");
const nucleiCount = document.querySelector("#nuclei-count");
const networkLength = document.querySelector("#network-length");
const branchPoints = document.querySelector("#branch-points");
const fragmentCount = document.querySelector("#fragment-count");
const tubulinCoverage = document.querySelector("#tubulin-coverage");

let selectedFile = null;
let previewUrl = null;
let isPredicting = false;

function setMessage(element, message) {
  element.textContent = message;
  element.hidden = !message;
}

function clearResult() {
  resultSection.hidden = true;
  predictedClass.textContent = "—";
  confidenceValue.textContent = "—";
  confidenceBar.style.width = "0";
  segmentationOverlay.removeAttribute("src");
  nucleiCount.textContent = "—";
  networkLength.textContent = "—";
  branchPoints.textContent = "—";
  fragmentCount.textContent = "—";
  tubulinCoverage.textContent = "—";
}

function resetPreview() {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  preview.removeAttribute("src");
  preview.hidden = true;
  previewEmpty.hidden = false;
  fileName.hidden = true;
  fileName.textContent = "";
}

function handleFileSelection(file) {
  setMessage(uploadError, "");
  setMessage(apiError, "");
  clearResult();

  if (!file) {
    selectedFile = null;
    resetPreview();
    predictButton.disabled = true;
    return;
  }

  if (!file.type.startsWith("image/")) {
    selectedFile = null;
    resetPreview();
    predictButton.disabled = true;
    setMessage(uploadError, "That file doesn't appear to be an image. Please choose an image file.");
    return;
  }

  selectedFile = file;
  resetPreview();
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.alt = `Preview of ${file.name}`;
  preview.hidden = false;
  previewEmpty.hidden = true;
  fileName.textContent = file.name;
  fileName.hidden = false;
  predictButton.disabled = false;
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0] ?? null;
  handleFileSelection(file);
  fileInput.value = "";
});

uploadArea.addEventListener("dragover", (event) => {
  event.preventDefault();
  uploadArea.classList.add("is-dragging");
});

uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("is-dragging");
});

uploadArea.addEventListener("drop", (event) => {
  event.preventDefault();
  uploadArea.classList.remove("is-dragging");
  handleFileSelection(event.dataTransfer?.files?.[0] ?? null);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setMessage(uploadError, "");
  setMessage(apiError, "");
  clearResult();

  if (!selectedFile) {
    setMessage(uploadError, "Choose an image before running a prediction.");
    fileInput.focus();
    return;
  }

  if (isPredicting) {
    return;
  }

  isPredicting = true;
  predictButton.disabled = true;
  predictButton.classList.add("is-loading");
  buttonLabel.textContent = "Predicting...";

  const formData = new FormData();
  formData.append(PREDICTION_API.fileField, selectedFile);

  try {
    const response = await fetch(PREDICTION_API.endpoint, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let message = `Prediction request failed (${response.status}). Please try again.`;
      try {
        const errorResult = await response.json();
        if (typeof errorResult.error === "string") {
          message = errorResult.error;
        }
      } catch {
        // Keep the status-based message when the API does not return JSON.
      }
      throw new Error(message);
    }

    let result;
    try {
      result = await response.json();
    } catch {
      throw new Error("The prediction service returned an unreadable response. Please try again.");
    }

    if (
      !result ||
      typeof result.class !== "string" ||
      !PREDICTION_API.classes.includes(result.class) ||
      typeof result.confidence !== "number" ||
      !Number.isFinite(result.confidence) ||
      result.confidence < 0 ||
      result.confidence > 1 ||
      typeof result.overlay !== "string" ||
      !result.overlay.startsWith("data:image/png;base64,") ||
      !Number.isSafeInteger(result.nuclei) ||
      result.nuclei < 0 ||
      !Number.isSafeInteger(result.network_length) ||
      result.network_length < 0 ||
      !Number.isSafeInteger(result.branch_points) ||
      result.branch_points < 0 ||
      !Number.isSafeInteger(result.fragments) ||
      result.fragments < 0 ||
      typeof result.tubulin_coverage !== "number" ||
      !Number.isFinite(result.tubulin_coverage) ||
      result.tubulin_coverage < 0 ||
      result.tubulin_coverage > 1
    ) {
      throw new Error("The prediction service returned an unexpected result. Please check the API response format.");
    }

    const percentage = result.confidence * 100;
    predictedClass.textContent = result.class;
    confidenceValue.textContent = `${percentage.toFixed(1)}%`;
    confidenceBar.style.width = `${percentage}%`;
    segmentationOverlay.src = result.overlay;
    nucleiCount.textContent = result.nuclei.toLocaleString();
    networkLength.textContent = result.network_length.toLocaleString();
    branchPoints.textContent = result.branch_points.toLocaleString();
    fragmentCount.textContent = result.fragments.toLocaleString();
    tubulinCoverage.textContent = `${(result.tubulin_coverage * 100).toFixed(1)}%`;
    resultSection.hidden = false;
  } catch (error) {
    setMessage(
      apiError,
      error instanceof TypeError
        ? location.protocol === "file:"
          ? "Open the demo at http://127.0.0.1:8000 after starting the local server with python app.py."
          : "Could not reach the prediction service. Check that the server is running and try again."
        : error.message || "The prediction could not be completed. Please try again.",
    );
  } finally {
    isPredicting = false;
    predictButton.disabled = !selectedFile;
    predictButton.classList.remove("is-loading");
    buttonLabel.textContent = "Predict image";
  }
});
