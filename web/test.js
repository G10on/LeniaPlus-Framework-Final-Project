

const body = document.getElementsByTagName("body"),
canvas = document.getElementById("map"),
ctx = canvas.getContext("2d"),
tracker = document.getElementById("tracker"),
ctx_tracker = tracker.getContext("2d"),
new_width = 800,
new_height = 800,
overlay = document.querySelector('#overlay'),
stream = canvas.captureStream(60),
chunks = [],
panelSelectorBtns = document.querySelectorAll('.panel-selector-btn'),
channelCheckbox = document.getElementById("checkboxes"),
channelSelector = document.querySelector('.channels-selector'),
playBtn = document.querySelector(".play-btn"),
nextStepBtn = document.querySelector(".next-step-btn"),
restartBtn = document.querySelector(".restart-btn"),
generateFromSeedBtn = document.querySelector(".generate-btn"),
saveVideoBtn = document.querySelector(".save-video-btn"),
saveStateBtn = document.querySelector(".save-state-btn"),
saveSample = document.querySelector(".save-sample"),
saveSampleName = document.querySelector(".save-sample-name"),
saveSampleNameBtn = document.querySelector(".save-sample-name-btn"),
exportBtn = document.querySelector(".export-btn"),
importBtn = document.querySelector(".import-btn"),
versionMenu = document.querySelector(".version-selector"),
versionLoadingTxt = document.querySelector(".version-loading-txt"),
stepNTxt = document.querySelector("#step-n-txt"),
closeBtn = document.querySelector(".close-btn"),
confirmExitWindow = document.getElementById("exit-confirmation-window"),
undoExitBtn = document.getElementById("undo-exit-btn"),
confirmExitBtn = document.getElementById("confirm-exit-btn"),
mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm" });


var rmsh = ['C', 'r', 'm', 's', 'h'],
Baw = ['B', 'a', 'w', 'T'],
is_playing = false,
frame_recorded = true,
expanded = false,
visibleChannels = [],
rowCount = 0,
tablePreview = document.getElementById("table-preview"),
tableKernel = document.getElementById("table-kernel"),
tableSamples = document.getElementById("table-samples");


canvas.width = new_width;
canvas.height = new_height;
stepNTxt.textContent = 0;


function addKernel() {
    // rowCount++;
    var row_preview = tablePreview.insertRow();
    var row_kernel = tableKernel.insertRow();
    // row_preview.id = rowCount;
    // row_kernel.id = rowCount;
    var rmshParamsCell = row_kernel.insertCell(0);
    var BParamsCell = row_kernel.insertCell(1);
    var aParamsCell = row_kernel.insertCell(2);
    var wParamsCell = row_kernel.insertCell(3);
    var TParamsCell = row_kernel.insertCell(4);
    // var addBInputCell = row.insertCell(4);
    var deleteRowCell = row_preview.insertCell(0);
    var previewCell = row_preview.insertCell(1);


    let kernelTitles = [rmsh.slice(1).join('')];
    kernelTitles = kernelTitles.concat(Baw);
    
    for (var i = 0; i < row_kernel.cells.length; i++) {

        let titleContainer = document.createElement("div");
        let inputList = document.createElement("div");
        titleContainer.className = "kernel-parameter-title parameter-title";
        inputList.className = "input-list";
        row_kernel.cells[i].appendChild(titleContainer);
        let newLbl = document.createElement("label");
        newLbl.textContent = kernelTitles[i];
        titleContainer.appendChild(newLbl);
        row_kernel.cells[i].appendChild(inputList);
    }




    addInputToKernel(row_preview, 0, rmsh[0], 0, false);
    
    for (var i = 1; i < rmsh.length; i++) {
        addInputToKernel(row_kernel, 0, rmsh[i], 0.2, false, innerDiv = true);
        // let inputContainer = document.createElement("div");
        // inputContainer.className = "inputContainer";
        // rmshParamsCell.appendChild(inputContainer);
        // let newLbl = document.createElement("label");
        // newLbl.textContent = rmsh[i];
        // let newInput = document.createElement("input");
        // newInput.type = "number";
        // // newInput.name = "input" + rowCount + "_" + i;
        // newInput.name = rmsh[i];
        // newInput.value = 0.2;
        // inputContainer.appendChild(newLbl);
        // inputContainer.appendChild(newInput);
    }

    // let inputContainer = document.createElement("div");
    // inputContainer.className = "inputContainer";
    // rmshParamsCell.appendChild(inputContainer);
    // let newLbl = document.createElement("label");
    // newLbl.textContent = 'B';
    // let newInput = document.createElement("input");
    // newInput.type = "number";
    // // newInput.name = "input" + rowCount + "_" + i;
    // newInput.name = 'B';
    // newInput.value = 0.2;
    // inputContainer.appendChild(newLbl);
    // inputContainer.appendChild(newInput);
    for (let k = 0; k < Baw.length; k++) {

        let addInputButton = document.createElement("button");
        addInputButton.className = "add-parameter-btn";
        addInputButton.type = "button";
        addInputButton.textContent = "Add " + Baw[k];
        addInputButton.onclick = function() { 
            addInput(this, k + 1, Baw[k], 0, true);
        }
        
        row_kernel.cells[k + 1].appendChild(addInputButton);
        
    }

    
    let deleteRowButton = document.createElement("button");
    deleteRowButton.type = "button";
    deleteRowButton.textContent = "Delete Kernel";
    deleteRowButton.onclick = function() { 
        deleteKernel(this);
     };
    deleteRowCell.appendChild(deleteRowButton);

    let EditRowButton = document.createElement("button");
    EditRowButton.type = "button";
    EditRowButton.textContent = "Edit Kernel";
    EditRowButton.onclick = function() { 
        displayKernelWindow(this);
     };
    previewCell.appendChild(EditRowButton);

    return [row_preview, row_kernel];
}

function deleteKernel(btn) {
    let row_preview = btn.parentNode.parentNode;
    let id = row_preview.rowIndex;
    row_preview.parentNode.removeChild(row_preview);
    tableKernel.deleteRow(id);
}

function addInputToKernel(row, cell_n, p, value, removable = true, innerDiv = false) {

    var inputList;
    if (innerDiv) {
        inputList = row.cells[cell_n].getElementsByClassName("input-list")[0];
    } else {
        inputList = row.cells[cell_n];
    }

    let inputContainer = document.createElement("div");
    inputContainer.className = "inputContainer";
    inputList.appendChild(inputContainer);
    let newInput = document.createElement("input");
    newInput.type = "number";
    // newInput.name = "input" + rowCount + "_" + i;
    newInput.name = p;
    newInput.value = value;
    if (p != '' && p) {
        let newLbl = document.createElement("label");
        newLbl.textContent = p;
        inputContainer.appendChild(newLbl);
    }
    inputContainer.appendChild(newInput);

    if (removable) {
        let removeButton = document.createElement("button");
        removeButton.className = "remove-btn";
        removeButton.type = "button";
        removeButton.innerHTML = "Delete";
        removeButton.onclick = function() { removeInput(this); };
        inputContainer.appendChild(removeButton);
    }

    return newInput;
}
  

function addInput(btn, cell_n, p, value, innerDiv) {
  var row = btn.parentNode.parentNode;
  addInputToKernel(row, cell_n, '', value, true, innerDiv);
}

function removeInput(btn) {
  var inputContainer = btn.parentNode;
  inputContainer.parentNode.removeChild(inputContainer);
//   var row = btn.parentNode.parentNode.parentNode;
//   var inputCount = row.cells[0].getElementsByClassName("inputContainer").length;
//   var inputs = row.cells[0].getElementsByTagName("input");
//   for (var i = 4; i < inputs.length; i++) {
//     inputs[i].name = "input" + rowCount + "_" + (i+1);
//   }
}


function displayKernelWindow(btn) {
    let row_preview = btn.parentNode.parentNode;
    let id = row_preview.rowIndex;
    let row_kernel = tableKernel.rows[id];

    let kernelWindow = document.querySelector(".kernel");
    kernelWindow.style.display = "block";
    row_kernel.style.display = "flex";
    overlay.style.display = "block";

    overlay.addEventListener('click', () => {
        kernelWindow.style.display = "none";
        row_kernel.style.display = "none";
        overlay.style.display = "none";
    });
}


function downloadFile(url, filename) {
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
}


async function updateWorldDisplay() {
    
    // let A = new Uint8ClampedArray(await eel.getWorld()());
    // let sz = Math.sqrt(A.length / 4);
    // let img_A = new ImageData(A, sz, sz);

    // ctx.createImageData(canvas.height, canvas.width);
    // ctx.clearRect(0, 0, canvas.width, canvas.height);
    // ctx.putImageData(img_A, 0, 0);

    var matrix = await eel.getWorld(visibleChannels)();
    // console.log(matrix.length);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (var i = 0; i < matrix.length; i++) {
        for (var j = 0; j < matrix[i].length; j++) {
            ctx.fillStyle = 'rgba(' + matrix[i][j].join(',') + ')';
            ctx.fillRect(i * new_width / matrix.length, j * new_height / matrix[i].length, new_width / matrix.length, new_height / matrix[i].length);
        }
    }

    var dataUri = canvas.toDataURL();

    let img = new Image();
    ctx.drawImage(img, 0, 0);
    img.src = dataUri;
    // getAnalysisFromPython();
    frame_recorded = false;
    // requestAnimationFrame(updateWorldDisplay);

}


async function getAnalysisFromPython() {
    
    let centers = await eel.getCoordinatesFromPython()();
    drawCenterMass(centers);
    drawDots(centers);
    updateStats();
    updateIndividualsButtons();
    // updateIndividualsStatWindows();
    updateAllIndividuals();
    // let coordinates = await eel.getCoordinatesFromPython()();
}

async function drawDots(coordinates) {

    const pointSize = 1;
    const canvasWidth = tracker.width;
    const canvasHeight = tracker.height;
  
    for (let id in coordinates) {
        ctx_tracker.fillStyle = getColorByKey(id);
        const [xPercent, yPercent] = coordinates[id];
        const x = xPercent * canvasWidth;
        const y = yPercent * canvasHeight;
        ctx_tracker.beginPath(); 
        ctx_tracker.arc(x, y, pointSize, 0, Math.PI * 2, true);
        ctx_tracker.fill(); 
    }
}



async function drawCenterMass(centers) {

    ctx.lineWidth = 4;

    for (let id in centers) {

        ctx.strokeStyle = getColorByKey(id);

        let x = centers[id][0];
        let y = centers[id][1];
        let detection_width = centers[id][2] * new_width;
        let detection_height = centers[id][3] * new_height;
        let half_width = detection_width / 2;
        let half_height = detection_height / 2;
        ctx.strokeRect(x * new_width - half_width, y * new_height - half_height, detection_width, detection_height);
    }

}

async function updateStats() {
    // updateGlobalSurvival();
    // updateGlobalreproduction(1);
    getNewStats(2);
    getNewStats(1);
    getNewStats(0);
}



function resetPlayer() {
    is_playing = false;
    stepNTxt.textContent = 0;
    playBtn.innerHTML = "&#9658";
    playBtn.style.background = "#4CAF50";
}


async function step() {
    stepNTxt.textContent = parseInt(stepNTxt.textContent) + 1;
    await eel.step()();
    await updateWorldDisplay();
    getAnalysisFromPython();
}

async function play() {
    while (is_playing) {
        await step();
    }
}


async function getParamsFromPython() {

    resetPlayer();
    versionLoadingTxt.innerText = "Compiling...";
    tablePreview.innerHTML = "";
    tableKernel.innerHTML = "";

    var data = await eel.getParameters()();

    // console.log(data)
    
    document.querySelector(".version-selector").value = data["version"];
    document.querySelector(".size").value = data["size"];
    document.querySelector(".num-channels").value = data["numChannels"];
    document.querySelector(".seed").value = data["seed"];
    document.querySelector(".theta").value = data["theta"];
    document.querySelector(".dd").value = data["dd"];
    document.querySelector(".dt").value = data["dt"];
    document.querySelector(".sigma").value = data["sigma"];

    for (var i = 0; i < data["r"].length; i++) {
        
        let [row_preview, row_kernel] = addKernel();
        rmsh.slice(1).forEach(k =>{
            
            // let inputs = row.cells[0].getElementsByTagName("input");
            let input = row_kernel.cells[0].getElementsByClassName("input-list")[0].querySelector("input[name='" + k + "']");
            input.value = data[k][i];
        })

        let input_C = row_preview.cells[0].querySelector("input[name='C']");
        input_C.value = data['C'][i];

        for (var k = 0; k < Baw.length - 1; k++) {

            for (var j = 0; j < data[Baw[k]][i].length; j++) {
                
                // let input = row.cells[0].querySelector("input[name='" + j + "']");
                let input = addInputToKernel(row_kernel, k + 1, '', data[Baw[k]][i][j], true, innerDiv = true);
                // input.value = data[Baw[k]][i][j];
                
            }
        }

                
    }

    var rows = tableKernel.rows;

    for (var c = 0; c < data['numChannels']; c++) {

        for (var j = 0; j < data['T'][c].length; j++) {

            input = addInputToKernel(rows[data['T'][c][j]], Baw.length, '', c, true, innerDiv = true);
            // input.value = c;
        }
    }

    generateCheckboxes(data['numChannels']);
    countSelectedCheckboxes();
    
    await updateWorldDisplay();
    versionLoadingTxt.innerText = "";
}


async function setParamsInPython(sampleName = null) {

    versionLoadingTxt.innerText = "Compiling...";
    ctx_tracker.clearRect(0, 0, tracker.width, tracker.height);
    let container = document.getElementById('individual-selector');
    container.innerHTML = "";



    let data = {};

    
    data["version"] = document.querySelector(".version-selector").value;
    data["size"] = parseInt(document.querySelector(".size").value);
    data["numChannels"] = parseInt(document.querySelector(".num-channels").value);
    data["seed"] = parseInt(document.querySelector(".seed").value);
    data["theta"] = parseFloat(document.querySelector(".theta").value);
    data["dd"] = parseInt(document.querySelector(".dd").value);
    data["dt"] = parseFloat(document.querySelector(".dt").value);
    data["sigma"] = parseFloat(document.querySelector(".sigma").value);

    // let kernelParmas = submitForm();

    for (var k = 0; k < rmsh.length; k++) {
        data[rmsh[k]] = [];
    }
    
    for (var k = 0; k < Baw.length; k++) {
        data[Baw[k]] = [];
    }

    for (var c = 0; c < data["numChannels"]; c++) {
        data['T'].push([]);
    }

    var rows_kernel = tableKernel.rows;
    var rows_preview = tablePreview.rows;

    for (var i = 0; i < rows_kernel.length; i++) {
        
        let row_kernel = rows_kernel[i];  
        let row_preview = rows_preview[i];  


        for (var k = 1; k < rmsh.length; k++) {
            
            // let inputs = row.cells[0].getElementsByTagName("input");
            let input = row_kernel.cells[0].getElementsByClassName("input-list")[0].querySelector("input[name='" + rmsh[k] + "']");
            data[rmsh[k]].push(parseFloat(input.value));
        }

        let input_C = row_preview.cells[0].querySelector("input[name='C']");
        let source_channel = Math.min(parseInt(input_C.value), data["numChannels"] - 1);
        data['C'].push(parseInt(source_channel));

        for (var k = 0; k < Baw.length - 1; k++) {

            // data[Baw[k]].push([]);
            // console.log(data);
            var inputs = row_kernel.cells[k + 1].getElementsByClassName("input-list")[0].getElementsByTagName("input");

            let B = [];

            for (var j = 0; j < inputs.length; j++) {
                
                // let input = row.cells[0].querySelector("input[name='" + j + "']");
                B.push(parseFloat(inputs[j].value));
                
            }
            data[Baw[k]].push(B);

        }
        
        var inputs = row_kernel.cells[Baw.length].getElementsByClassName("input-list")[0].getElementsByTagName("input");


        for (var j = 0; j < inputs.length; j++) {
            let target_channel = Math.min(inputs[j].value, data["numChannels"] - 1);
            data['T'][target_channel].push(parseInt(i));
        }
    }

    
    
    if (sampleName != null) {
        await eel.setSample(sampleName, data)();
    } else {
        await eel.setParameters(data)();
    }
    
    await getParamsFromPython();
    // await updateWorldDisplay();
    versionLoadingTxt.innerText = "";
}

// TODO: REDUCE DUPLICATE CODE FROM SETPARAMSINPYTHON FUNCTION
async function generateKernelParamsInPython() {

    is_playing = false;
    versionLoadingTxt.innerText = "Compiling...";

    let data = {};
    
    data["version"] = document.querySelector(".version-selector").value;
    data["size"] = parseInt(document.querySelector(".size").value);
    data["numChannels"] = parseInt(document.querySelector(".num-channels").value);
    data["seed"] = parseInt(document.querySelector(".seed").value);
    data["theta"] = parseFloat(document.querySelector(".theta").value);
    data["dd"] = parseInt(document.querySelector(".dd").value);
    data["dt"] = parseFloat(document.querySelector(".dt").value);
    data["sigma"] = parseFloat(document.querySelector(".sigma").value);

    // let kernelParmas = submitForm();

    for (var k = 0; k < rmsh.length; k++) {
        data[rmsh[k]] = [];
    }
    
    for (var k = 0; k < Baw.length; k++) {
        data[Baw[k]] = [];
    }

    var rows_kernel = tableKernel.rows;
    var rows_preview = tablePreview.rows;

    for (var i = 0; i < rows_kernel.length; i++) {
        
        let row_kernel = rows_kernel[i];
        let row_preview = rows_preview[i];


        rmsh.slice(1).forEach(k =>{
            
            // let inputs = row.cells[0].getElementsByTagName("input");
            let input = row_kernel.cells[0].querySelector("input[name='" + k + "']");
            data[k].push(parseFloat(input.value));
        })

        for (var k = 0; k < Baw.length; k++) {

            // data[Baw[k]].push([]);
            // console.log(data);
            inputs = row_kernel.cells[k + 1].getElementsByTagName("input");

            let B = [];

            for (var j = 0; j < inputs.length; j++) {
                
                // let input = row.cells[0].querySelector("input[name='" + j + "']");
                B.push(parseFloat(inputs[j].value));
                
            }
            data[Baw[k]].push(B);

        }
    }
    
    await eel.generateKernel(data)();
    // await updateWorldDisplay();
    await getParamsFromPython();
    versionLoadingTxt.innerText = "";
}


function panelChanger(event) {
    let divs = document.querySelectorAll('.panel-section');
    divs.forEach(div => {
        if (div.id === event.target.id + '-content') {
            div.style.display = 'flex';
        } else {
            div.style.display = 'none';
        }
    });

    // Use SWITCH!!!
    if (event.target.id === "samples") {
        loadSamples();
    }
}


function addSample (sampleName) {
    
    // create a new div element for each image
    let sampleContainer = document.createElement('div');
    sampleContainer.className = 'sample-container'; // assign a class to style with CSS
    // document.body.appendChild(sampleContainer); // add the div to the body of the HTML document
    sampleContainer.id = sampleName;

    // add an img element inside the div
    let sampleImg = document.createElement('img');
    sampleImg.className = 'sample-img'; // assign a class to style with CSS
    sampleImg.setAttribute('data-src', sampleName);
    sampleImg.src = "images/samples/" + sampleName + ".png"; // set the src attribute to the current image source
    sampleImg.addEventListener('click', function() {
        let imageSource = this.getAttribute('data-src');
        setParamsInPython(imageSource);
    });
    sampleContainer.appendChild(sampleImg); // add the img to the div

    // add a label inside the div
    let sampleLbl = document.createElement('label');
    sampleLbl.className = 'sample-lbl'; // assign a class to style with CSS
    sampleLbl.innerHTML = sampleName; // set the label text
    sampleContainer.appendChild(sampleLbl); // add the label to the div

    let deleteSampleButton = document.createElement("button");
    deleteSampleButton.type = "button";
    deleteSampleButton.className = "delete-sample-btn";
    // deleteSampleButton.innerHTML = ;
    deleteSampleButton.textContent = "DELETE SAMPLE";
    deleteSampleButton.onclick = async function() { 
        await deleteSample(this);
    };
    sampleContainer.appendChild(deleteSampleButton);

    return sampleContainer;
    
}

async function deleteSample(btn) {
    let sampleContainer = btn.parentNode;
    await eel.deleteSample(sampleContainer.id)();
    sampleContainer.style.display = "none";
}



async function loadSamples () {
    
    var sampleList = document.getElementById("sample-list");
    sampleList.innerHTML = "";

    let sampleNames = await eel.getSampleNames()();
    
    for (var i = 0; i < sampleNames.length; i++) {
        
        let sampleContainer = addSample(sampleNames[i]);
        sampleList.appendChild(sampleContainer);
        
    }
}
    

function setChannelSelector () {
    
    let options = ['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5'];
    options.forEach((option, index) => {
        const optionElement = document.createElement('option');
        optionElement.textContent = option;
        optionElement.value = index;
        channelSelector.appendChild(optionElement);
    });

}

function displaySaveNameWindow(btn) {
    
    saveSample.style.display = "flex";
    overlay.style.display = "block";

    overlay.addEventListener('click', () => {
        saveSample.style.display = "none";
        overlay.style.display = "none";
    });
}



// function clearCheckboxes() {
//   var checkboxesDiv = document.getElementById("checkboxes");
//   while (checkboxesDiv.firstChild) {
//     checkboxesDiv.removeChild(checkboxesDiv.firstChild);
//   }
// }

function generateCheckboxes(numChannels) {
    var checkboxesDiv = document.getElementById("checkboxes");
    //   clearCheckboxes();
    checkboxesDiv.innerHTML = "";
    for (let index = 0; index < numChannels; index++) {
        var label = document.createElement('label');
        var checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `channel-${index}`;
        let maxChannels = Math.min(numChannels, 3);
        if (index < maxChannels) {checkbox.checked = true;}
        label.appendChild(checkbox);
        let opt_txt = document.createTextNode(`channel ${index}`);
        label.appendChild(opt_txt);
        checkboxesDiv.appendChild(label);
    }
}


function showCheckboxes() {
  var checkboxes = document.getElementById("checkboxes");
  if (!expanded) {
    checkboxes.style.display = "block";
    expanded = true;
  } else {
    checkboxes.style.display = "none";
    expanded = false;
  }
}

function countSelectedCheckboxes() {
    var checkboxes = document.querySelectorAll('#checkboxes input[type="checkbox"]');
    let numChannels = document.querySelector(".num-channels").value;
    //   var selectedCount = 0;
    let maxChannels = Math.min(numChannels, 3);
    visibleChannels = [];
    checkboxes.forEach(function(checkbox, index) {
        if (checkbox.checked) {
        //   selectedCount++;
        visibleChannels.push(index);
        }
    });
    if (visibleChannels.length >= maxChannels) {
        checkboxes.forEach(function(checkbox) {
        if (!checkbox.checked) {
            checkbox.disabled = true;
        }
        });
    } else {
        checkboxes.forEach(function(checkbox) {
        checkbox.disabled = false;
        });
    }
    if (!is_playing) { updateWorldDisplay()}
}


function openNewWindow() {
    window.open("tracker.html", "TEST_New_Window", "width=600,height=400");
}
eel.expose(openNewWindow);





panelSelectorBtns.forEach(button => {
  button.addEventListener('click', panelChanger);
});

playBtn.addEventListener("click", () => {

    is_playing = !is_playing;

    if (is_playing) {
        play();
        playBtn.innerHTML = "&#10074;&#10074";
        playBtn.style.background = "tomato";
    } else {
        playBtn.innerHTML = "&#9658";
        playBtn.style.background = "#4CAF50";
    }
    
})

nextStepBtn.addEventListener("click", async () => {

    is_playing = false;
    step();
    playBtn.innerHTML = "&#9658";
    playBtn.style.background = "#4CAF50";
    
})


restartBtn.addEventListener("click", () => {
    
    
    setParamsInPython();
    // getKernelParamsFromWeb();
})


generateFromSeedBtn.addEventListener("click", () => {
    
    generateKernelParamsInPython();
    // getKernelParamsFromWeb();
})

saveVideoBtn.addEventListener("click", () => {

    let nSteps = parseInt(document.querySelector(".sec-to-save").value);
    // eel.saveNStepsToVideo(nSteps);
    mediaRecorder.start();
    setTimeout(function() {
        mediaRecorder.stop();
    }, nSteps * 1000);
})


saveStateBtn.addEventListener("click", () => {
    
    displaySaveNameWindow();
    
})

saveSampleNameBtn.addEventListener("click", () => {
    
    let name = saveSampleName.value;
    saveSampleName.value = "";
    // Convert the canvas image data to a base64-encoded string
    let imageData = canvas.toDataURL("image/png");
    
    eel.saveParameterState(imageData, name)();

    saveSample.style.display = "none";
    overlay.style.display = "none";
})

exportBtn.addEventListener("click", () => {
    eel.saveParameterState()();
})

importBtn.addEventListener("click", async () => {
    
    await eel.loadParameterState()();
    getParamsFromPython();
    await updateWorldDisplay();
})

mediaRecorder.ondataavailable = function(e) {
    if (!frame_recorded) {
        chunks.push(e.data);
        frame_recorded = true;
    }
};

mediaRecorder.onstop = function() {
  const blob = new Blob(chunks, { type: "video/webm" });
  const url = URL.createObjectURL(blob);
  downloadFile(url, "myVideo.webm");
};

channelCheckbox.addEventListener("change", countSelectedCheckboxes);

closeBtn.addEventListener("click", async () => {
    confirmExitWindow.style.display = "flex";
    overlay.style.display = "block";

    overlay.addEventListener('click', () => {
        confirmExitWindow.style.display = "none";
        overlay.style.display = "none";
    });
})

undoExitBtn.addEventListener("click", async () => {
    confirmExitWindow.style.display = "none";
    overlay.style.display = "none";
})

confirmExitBtn.addEventListener("click", async () => {
    await eel.shutdown();
    window.close();
})





setChannelSelector();

getParamsFromPython();

getAnalysisFromPython();






