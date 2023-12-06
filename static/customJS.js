const config = window.factorConfig;

const getRiskCategoryMessage = (riskCategory) => {
  const riskCategoryHTMLMessages = {
    PROFILE_1: `Risk profile: <b><span style="color: red;">High risk</span></b>`,
    PROFILE_2: `Risk profile: <b><span style="color: green;">Low risk</span></b>`,
  };

  return riskCategoryHTMLMessages[riskCategory];
};

const getSummaryItemHTML = ({ description, label, value, id }) => `
              <li id="summary${id}" class="list-group-item d-flex justify-content-between lh-condensed">
                <div>
                  <h6 class="my-0">${label}</h6>
                  ${
                    description
                      ? `<small class='text-muted'>${description}</small>`
                      : ""
                  }
                </div>
                <span class="text-muted" id="summaryVal${id}">${value}</span>
              </li>`;

const getItemHTML = ({ id, min, max, label, step, value }) => `
      <div class="form-group mb-4" id="container${id}">
        <label for="${id}"
          >${label}<span class="text-muted">(min: ${min}, max: ${max})</span>
          <span data-toggle="tooltip" title="Remove this factor" data-placement="top" class="remove-feature" onclick="removeFeature(${id})">⛔️</span>
          <span class="info-feature" onclick="showInfo(${id})">?</span>
        </label>
        <div class="row">
          <div class="col-9 d-flex">
            <input
              id="${id}"
              min="${min}"
              max="${max}"
              type="range"
              step="${step}"
              value="${value}"
              class="form-control-range factors-input"
              oninput="handleRangeChange(this);" />
          </div>

          <div class="col-3 d-flex">
            <input
              min="${min}"
              max="${max}"
              type="number"
              step="${step}"
              value="${value}"
              required
              id="${id}_current"
              class="form-control factors-input factors-input-box"
              onchange="handleRangeInput(this, ${id});" />
          </div>
        </div>
      </div>`;

const getInfoModalBody = ({ info, img }) => `<p>${info}</p>
      <img src="${img}" class="info-img"/>`;

const getResultModalBody = ({ riskCategory, confidenceScore }) => `
      <div class="my-3 row align-items-center">
        <div class="col-9 result-text-display">
          <div id="resultText">${getRiskCategoryMessage(riskCategory)}</div>
          <p>Confidence Score: ${confidenceScore}%</p>
        </div>
        <div class="col-3">
          <div id="resultProgressBar" role="progressbar" style="--value: ${confidenceScore}"></div>
        </div>
      </div>
      <p class="my-1 mt-5">
        The <b>confidence score</b> represents the level of certainty in the risk
        category prediction. Higher scores indicate a higher level of confidence in
        the prediction
      </p>
      <p class="text-muted my-1">
        Disclaimer: This risk category prediction is based on your input parameters
        and is a result of an academic research project. It should not be considered a
        substitute for professional medical advice. Please consult with a healthcare
        professional for personalized medical guidance.
      </p>
`;

const showSWSError = ({ message }) => {
  const errorMsgEl = document.querySelector(
    "label.text-danger.small.font-weight-bold.error-helper"
  );
  errorMsgEl.innerText = message;
  document.querySelector(".selectpicker-wrp").classList.add("error-show");
};

const removeSWSError = () => {
  document.querySelector(".selectpicker-wrp").classList.remove("error-show");
};

const addNewFactor = (event) => {
  var selectedItemValue = $(".selected-item")[0].innerText;
  console.log(selectedItemValue);
  if (!selectedItemValue || selectedItemValue === "Select a test or symptom ") {
    showSWSError({ message: "Please select a value from the dropdown" });
  } else {
    removeSWSError();

    const selectedConfigItemKey = Object.keys(config).find(
      (configItemKey) => config[configItemKey].label === selectedItemValue
    );

    if (
      document.querySelectorAll(`#container${selectedConfigItemKey}`).length
    ) {
      showSWSError({
        message: "This value already exists",
      });
      return;
    }

    const formHTMLToBeInserted = getItemHTML({
      id: selectedConfigItemKey,
      ...config[selectedConfigItemKey],
    });
    const summaryHTMLToBeInserted = getSummaryItemHTML({
      id: selectedConfigItemKey,
      ...config[selectedConfigItemKey],
    });

    const topLevelElement = document.getElementById("swsTopEl");
    $(formHTMLToBeInserted).insertAfter(topLevelElement);

    const summaryULContainer = document.getElementById("listOfRisks");
    $(summaryHTMLToBeInserted).insertBefore(summaryULContainer);
    updateCountOnFactorsChange();
    $('[data-toggle="tooltip"]').tooltip({ container: "body" });
  }
};

const handleRangeChange = (props) => {
  const currentInputElement = document.querySelector(`#${props.id}_current`);
  currentInputElement.value = props.value;

  const summaryULContainer = document.getElementById(`summaryVal${props.id}`);

  summaryULContainer.innerText = props.value;
};

$(document).ready(function () {
  console.log("ready!");
  console.log($("#swsFactors"));
  // <option data-subtext="Two">Two</option>

  $('[data-toggle="tooltip"]').tooltip({ container: "body" });

  if (window.CSS && window.CSS.registerProperty) {
    window.CSS.registerProperty({
      name: "--percentage",
      syntax: "<number>",
      inherits: true,
      initialValue: 0,
    });
  }

  const wrapperElement = new DocumentFragment();

  for (item of Object.values(config)) {
    const optionElement = document.createElement("div");
    optionElement.classList.add("select-item");
    optionElement.innerText = item.label;
    wrapperElement.append(optionElement);
  }
  $("#selectOptionsContainer").append(wrapperElement);
  setTimeout(() => {
    registerSelectListeners();
    registerResetFormListener();
    setContainerHeightObserver();
  }, 10);
});

const registerResetFormListener = () => {
  document.getElementById("resetButton").addEventListener("click", resetForm);
};
const resetSelectedOption = () => {};

const registerSelectListeners = () => {
  document
    .querySelector(".selected-item")
    .addEventListener("click", function () {
      this.nextElementSibling.classList.toggle("hidden");
      this.nextElementSibling.nextElementSibling.classList.toggle("hidden");
    });

  document.querySelectorAll(".select-item").forEach((item) => {
    item.addEventListener("click", function () {
      let selectedItem = this.parentNode.previousElementSibling;
      selectedItem.textContent = this.textContent;
      this.parentNode.classList.add("hidden");
      this.parentNode.nextElementSibling.classList.add("hidden");
    });
  });

  document
    .querySelector(".select-search")
    .addEventListener("keyup", function () {
      let filter = this.value.toUpperCase();
      let items = this.previousElementSibling.children;
      for (let i = 0; i < items.length; i++) {
        let txtValue = items[i].textContent || items[i].innerText;
        if (txtValue.toUpperCase().indexOf(filter) > -1) {
          items[i].style.display = "";
        } else {
          items[i].style.display = "none";
        }
      }
    });

  document.addEventListener("click", function (event) {
    var selectBox = document.querySelector(".select-box");
    if (!selectBox.contains(event.target)) {
      var selectItems = document.querySelector(".select-items");
      var selectSearch = document.querySelector(".select-search");
      selectItems.classList.add("hidden");
      selectSearch.classList.add("hidden");
    }
  });
};

const resetForm = (event) => {
  event.preventDefault();
  document.getElementById("pdpForm").reset();
  clickAllRemoveFeature();
  removeSWSError();
  $(".selected-item")[0].innerText = "Select a test or symptom ";
};

const clickAllRemoveFeature = () => {
  var elements = document.querySelectorAll(".remove-feature");
  elements.forEach(function (element) {
    element.click();
  });
};

var validateAndGetFormData = () => {
  var genderEl = document.querySelector('[name="gender"]:checked');
  var ageEl = document.querySelector("#Age");
  var factorsInputArr = document.querySelectorAll(
    "input.form-control-range.factors-input"
  );

  if (!genderEl || !genderEl.value) {
    throw new Error("Please input gender");
  }
  if (!ageEl || !ageEl.value) {
    throw new Error("Please input age");
  }

  if (factorsInputArr.length < 5) {
    throw new Error("Please select at least 5 features to continue");
  }
  var genderValue = genderEl.value.toLowerCase() === "male" ? 0 : 1;

  // Gender=Male&Age=22&num_features=3&Feature1=Anxiety&Score_Feature1=3
  var formData = new FormData();
  formData.append("Gender", genderValue);
  formData.append("Age", ageEl.value);
  formData.append("num_features", factorsInputArr.length);

  factorsInputArr.forEach((item, index) => {
    formData.append(`Feature${index + 1}`, item.id);
    formData.append(`Score_Feature${index + 1}`, item.value);
  });

  return formData;
};

const showLoadingOnSubmitCTA = () => {
  const loaderModal = document.getElementById("fullPageLoader");
  const submitFormCTAEl = document.getElementById("submitFormCTA");
  loaderModal.classList.add("d-flex");
  submitFormCTAEl.innerText = "Predicting using some AI Magic";
  submitFormCTAEl.classList.add("disabled");
};

$("#pdpForm").bind("submit", async function (e) {
  e.preventDefault();

  try {
    var formData = validateAndGetFormData();
    showLoadingOnSubmitCTA();
    let response = await fetch("/submit", {
      method: "POST",
      body: formData,
    });
    const result = await response.json();

    var confidenceScore = parseInt(result.confidenceScore);
    var riskCategory = result.userStatus;
    document.getElementById("resultModalLabel").innerText =
      "Prediction results";

    const bodyEl = document.getElementById("resultModalBody");
    bodyEl.innerHTML = getResultModalBody({
      confidenceScore,
      riskCategory,
    });

    $("#resultModal").modal();
  } catch (error) {
    resetSubmitButton();
    alert(error);
  }
});

const resetSubmitButton = () => {
  const loaderModal = document.getElementById("fullPageLoader");
  loaderModal.classList.remove("d-flex");

  const submitFormCTAEl = document.getElementById("submitFormCTA");
  submitFormCTAEl.innerText = "Calculate my risk profile";
  submitFormCTAEl.classList.remove("disabled");
};

// Handle modal close
$("#resultModal").on("hide.bs.modal", resetSubmitButton);

const updateCountOnFactorsChange = () => {
  const factorsInputArr = document.querySelectorAll(
    "input.form-control-range.factors-input"
  );

  const factorsCounterEl = document.querySelector(
    "span.badge.badge-secondary.badge-pill.factors-count"
  );

  factorsCounterEl.innerText = factorsInputArr.length;
};

const handleRangeInput = (event, { id }) => {
  if (event.stopPropagation) {
    event.stopPropagation();
    event.preventDefault();
  } else if (window.event) {
    window.event.cancelBubble = true;
  }

  const { min, max } = config[id];

  if (Number(event.value) > Number(max)) {
    alert("Input out of range");
    setTimeout(() => {
      event.value = max;
    }, 10);
  }

  if (Number(event.value) < Number(min)) {
    alert("Input out of range");
    setTimeout(() => {
      event.value = min;
    }, 10);
  }

  const rangeSliderEl = document.getElementById(id);
  rangeSliderEl.value = event.value;

  const summaryULContainer = document.getElementById(`summaryVal${id}`);
  summaryULContainer.innerText = event.value;
};

const stopOnEnter = (e) => {
  //Enter key
  if (e && (e.which == 13 || e.keyCode == 13)) {
    return false;
  }
};

$("#pdpForm").keypress(stopOnEnter);

const removeFeature = (element) => {
  /* eslint-disable no-debugger, no-console */
  const containerId = `container${element.id}`;
  const summaryId = `summary${element.id}`;
  console.log("id", containerId);
  const itemToRemove = document.getElementById(containerId);
  const summaryItemToRemove = document.getElementById(summaryId);
  itemToRemove.remove();
  summaryItemToRemove.remove();
  updateCountOnFactorsChange();
  $(".tooltip").hide();
};

const showInfo = (element) => {
  const clickedItem = element.id;

  const headerEl = document.getElementById("resultModalLabel");
  headerEl.innerText = factorConfig[clickedItem].label;

  const bodyEl = document.getElementById("resultModalBody");
  bodyEl.innerHTML = getInfoModalBody({ ...factorConfig[clickedItem] });

  $("#resultModal").modal();
};

const showDataModal = (element) => {
  const infoToShow = element.dataset["info"];
  const headerToShow = element.innerText;

  const headerEl = document.getElementById("resultModalLabel");
  headerEl.innerHTML = headerToShow;

  const bodyEl = document.getElementById("resultModalBody");
  bodyEl.innerHTML = infoToShow;

  $("#resultModal").modal();
};

const setContainerHeightObserver = () => {
  window.addEventListener("resize", setMaxHeight);

  function setMaxHeight() {
    var viewportHeight = window.innerHeight;
    var selectedItemsRect = document
      .querySelector(".selected-item")
      .getBoundingClientRect();
    var maxHeight = Math.ceil(viewportHeight - selectedItemsRect.bottom - 60);
    document.getElementById("selectOptionsContainer").style.maxHeight =
      maxHeight + "px";
  }

  // Call the function initially
  setMaxHeight();
};
