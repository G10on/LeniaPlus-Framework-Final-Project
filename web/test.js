

const canvas = document.getElementById("map"),
ctx = canvas.getContext("2d"),
new_width = 800,
new_height = 800,
overlay = document.querySelector('#overlay'),
stream = canvas.captureStream(),
chunks = [],
panelSelectorBtns = document.querySelectorAll('.panel-selector-btn'),
channelSelector = document.querySelector('.channels-selector'),
playBtn = document.querySelector(".play-btn"),
nextStepBtn = document.querySelector(".next-step-btn"),
restartBtn = document.querySelector(".restart-btn"),
generateFromSeedBtn = document.querySelector(".generate-btn"),
saveVideoBtn = document.querySelector(".save-video-btn"),
saveStateBtn = document.querySelector(".save-state-btn"),
loadStateBtn = document.querySelector(".load-state-btn"),
versionMenu = document.querySelector(".version-selector"),
versionLoadingTxt = document.querySelector(".version-loading-txt"),
stepNTxt = document.querySelector("#step-n-txt"),
mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm" });


var rmsh = ['C', 'r', 'm', 's', 'h'],
Baw = ['B', 'a', 'w', 'T'],
is_playing = false,
rowCount = 0,
tablePreview = document.getElementById("table-preview"),
tableKernel = document.getElementById("table-kernel"),
tableSamples = document.getElementById("table-samples");


canvas.width = new_width;
canvas.height = new_height;
stepNTxt.textContent = 0;


function addKernel() {
    // rowCount++;
    row_preview = tablePreview.insertRow();
    row_kernel = tableKernel.insertRow();
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
    if (p != '') {
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

    let kernelWindow = document.querySelector(".kernel-window");
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

    var matrix = await eel.getWorld()();
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

    console.log(data)
    
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
    
    await updateWorldDisplay();
    versionLoadingTxt.innerText = "";
}


async function setParamsInPython(sampleName = null) {

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
        data['C'].push(parseInt(input_C.value));

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
            data['T'][inputs[j].value].push(parseInt(i));
        }
    }

    
    
    if (sampleName != null) {
        await eel.setSample(sampleName, data)();
    } else {
        await eel.setParameters(data)();
    }
    
    await getParamsFromPython();
    await updateWorldDisplay();
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
    await updateWorldDisplay();
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
    var sampleContainer = document.createElement('div');
    sampleContainer.className = 'sample-container'; // assign a class to style with CSS
    document.body.appendChild(sampleContainer); // add the div to the body of the HTML document

    // add an img element inside the div
    var sampleImg = document.createElement('img');
    sampleImg.className = 'sample-img'; // assign a class to style with CSS
    sampleImg.setAttribute('data-src', sampleName);
    sampleImg.src = "./images/" + sampleName + ".png"; // set the src attribute to the current image source
    sampleContainer.appendChild(sampleImg); // add the img to the div

    // add a label inside the div
    var sampleLbl = document.createElement('label');
    sampleLbl.className = 'sample-lbl'; // assign a class to style with CSS
    sampleLbl.innerHTML = sampleName; // set the label text
    sampleContainer.appendChild(sampleLbl); // add the label to the div

    sampleContainer.addEventListener('click', function() {
        let imageSource = this.querySelector('.sample-img').getAttribute('data-src');
        setParamsInPython(imageSource);
    });

    return sampleContainer;
    
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

$('.channels-selector').on('change', function() {
  if ($('select option:selected').length > 3) {
    $(this).find('option:selected').removeAttr('selected');
    console.log("No way")
  }
});






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
    eel.saveParameterState()();
})

loadStateBtn.addEventListener("click", async () => {
    
    await eel.loadParameterState()();
    getParamsFromPython();
    await updateWorldDisplay();
})

mediaRecorder.ondataavailable = function(e) {
  chunks.push(e.data);
};

mediaRecorder.onstop = function() {
  const blob = new Blob(chunks, { type: "video/webm" });
  const url = URL.createObjectURL(blob);
  downloadFile(url, "myVideo.webm");
};









setChannelSelector();

getParamsFromPython();






