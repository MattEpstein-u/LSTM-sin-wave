let generatedWaves = [];
let model;
let lossHistory = [];

document.getElementById('generateBtn').addEventListener('click', generateWaves);
document.getElementById('displayBtn').addEventListener('click', plotWaves);
document.getElementById('trainBtn').addEventListener('click', trainLSTM);

// Auto-run on load
window.onload = function() {
    generateWaves();
    plotWaves();
};

async function trainLSTM() {
    if (generatedWaves.length === 0) {
        alert("Generate data first!");
        return;
    }
    
    const statusDiv = document.getElementById('trainingStatus');
    statusDiv.innerText = "Preparing data...";
    await tf.nextFrame();
    
    // Prepare data
    // X: first 49 points' y-values
    // y: 50th point's y-value
    const inputs = [];
    const labels = [];
    
    for (let i = 0; i < generatedWaves.length; i++) {
        const wave = generatedWaves[i];
        const waveInputs = [];
        for (let j = 0; j < 49; j++) {
            waveInputs.push(wave[j].y);
        }
        inputs.push(waveInputs);
        labels.push(wave[49].y);
    }

    // Generate validation data
    const minAmp = parseFloat(document.getElementById('minAmp').value);
    const maxAmp = parseFloat(document.getElementById('maxAmp').value);
    const minWave = parseFloat(document.getElementById('minWave').value);
    const maxWave = parseFloat(document.getElementById('maxWave').value);
    const negProb = parseFloat(document.getElementById('negProb').value);
    
    const valCount = Math.max(20, Math.floor(generatedWaves.length * 0.2));
    const valData = generateSinWaves(valCount, minAmp, maxAmp, minWave, maxWave, negProb);
    
    const valInputs = [];
    const valLabels = [];
    for (let i = 0; i < valData.length; i++) {
        const wave = valData[i];
        const waveInputs = [];
        for (let j = 0; j < 49; j++) {
            waveInputs.push(wave[j].y);
        }
        valInputs.push(waveInputs);
        valLabels.push(wave[49].y);
    }

    statusDiv.innerText = "Creating tensors...";
    await tf.nextFrame();
    
    let xs, ys, valXs, valYs;
    try {
        // Create 2D tensor first [samples, timeSteps] then expand to [samples, timeSteps, features]
        // This is more robust than trying to create 3D tensor directly from 2D array
        xs = tf.tensor2d(inputs, [inputs.length, 49]).expandDims(2);
        ys = tf.tensor2d(labels, [labels.length, 1]);
        
        valXs = tf.tensor2d(valInputs, [valInputs.length, 49]).expandDims(2);
        valYs = tf.tensor2d(valLabels, [valLabels.length, 1]);
    } catch (e) {
        statusDiv.innerText = "Error creating tensors: " + e.message;
        console.error(e);
        return;
    }
    
    statusDiv.innerText = "Compiling model...";
    await tf.nextFrame();

    // Define model
    model = tf.sequential();
    model.add(tf.layers.lstm({
        units: 50,
        inputShape: [49, 1],
        returnSequences: false
    }));
    model.add(tf.layers.dense({units: 1}));
    
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'meanSquaredError'
    });
    
    lossHistory = [];
    statusDiv.innerText = "Training started...";
    await tf.nextFrame();
    
    let currentEpoch = 0;
    const totalEpochs = 20;

    await model.fit(xs, ys, {
        epochs: totalEpochs,
        batchSize: 32,
        validationData: [valXs, valYs],
        callbacks: {
            onEpochBegin: async (epoch, logs) => {
                currentEpoch = epoch + 1;
                statusDiv.innerText = `Epoch: ${currentEpoch}/${totalEpochs} - Starting...`;
                await tf.nextFrame();
            },
            onBatchEnd: async (batch, logs) => {
                statusDiv.innerText = `Epoch: ${currentEpoch}/${totalEpochs} - Batch: ${batch} - Loss: ${logs.loss.toFixed(6)}`;
                await tf.nextFrame();
            },
            onEpochEnd: async (epoch, logs) => {
                lossHistory.push({
                    epoch: epoch + 1, 
                    loss: logs.loss,
                    val_loss: logs.val_loss
                });
                drawLossGraph();
                statusDiv.innerText = `Epoch: ${epoch + 1}/${totalEpochs} - Loss: ${logs.loss.toFixed(6)} - Val Loss: ${logs.val_loss.toFixed(6)}`;
                await tf.nextFrame();
            }
        }
    });
    
    statusDiv.innerText = "Training complete!";
    xs.dispose();
    ys.dispose();
    valXs.dispose();
    valYs.dispose();

    await evaluateLSTM();
}

async function evaluateLSTM() {
    const statusDiv = document.getElementById('trainingStatus');
    statusDiv.innerText = "Evaluating model...";
    
    // Generate test data
    const minAmp = parseFloat(document.getElementById('minAmp').value);
    const maxAmp = parseFloat(document.getElementById('maxAmp').value);
    const minWave = parseFloat(document.getElementById('minWave').value);
    const maxWave = parseFloat(document.getElementById('maxWave').value);
    const negProb = parseFloat(document.getElementById('negProb').value);
    
    const testCount = 5;
    const testData = generateSinWaves(testCount, minAmp, maxAmp, minWave, maxWave, negProb);
    
    const testInputs = [];
    const testLabels = [];
    for (let i = 0; i < testData.length; i++) {
        const wave = testData[i];
        const waveInputs = [];
        for (let j = 0; j < 49; j++) {
            waveInputs.push(wave[j].y);
        }
        testInputs.push(waveInputs);
        testLabels.push(wave[49].y);
    }
    
    const testXs = tf.tensor2d(testInputs, [testInputs.length, 49]).expandDims(2);
    const preds = model.predict(testXs);
    const predValues = await preds.data();
    
    testXs.dispose();
    preds.dispose();
    
    // Draw evaluation
    const canvas = document.getElementById('evalCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Zoom settings: Show last 15 points (approx last 30% of the wave)
    const pointsToShow = 15;
    const totalPoints = 50;
    const startIndex = totalPoints - pointsToShow; // e.g. 35
    
    // Calculate X range for zoom
    // x goes from 0 to 1.
    // x at startIndex = startIndex / 49
    const minXVal = startIndex / 49;
    const maxXVal = 1.0; 

    // Find min and max y for scaling ONLY within the zoomed region
    let minY = Infinity;
    let maxY = -Infinity;
    
    // Check test data points in range
    testData.forEach(wave => {
        for(let j=startIndex; j<50; j++) {
             const p = wave[j];
             minY = Math.min(minY, p.y);
             maxY = Math.max(maxY, p.y);
        }
    });
    // Check predicted values
    for(let i=0; i<predValues.length; i++) {
        minY = Math.min(minY, predValues[i]);
        maxY = Math.max(maxY, predValues[i]);
    }

    const yRange = maxY - minY;
    if (yRange === 0) {
        minY -= 1;
        maxY += 1;
    } else {
        minY -= yRange * 0.1;
        maxY += yRange * 0.1;
    }

    const marginTop = 20;
    const marginBottom = 40;
    const marginLeft = 60;
    const marginRight = 20;

    const plotWidth = canvas.width - marginLeft - marginRight;
    const plotHeight = canvas.height - marginTop - marginBottom;

    // Add a little padding to X view
    const xPadding = (maxXVal - minXVal) * 0.05;
    const viewMinX = minXVal;
    const viewMaxX = maxXVal + xPadding;

    const mapX = (x) => marginLeft + ((x - viewMinX) / (viewMaxX - viewMinX)) * plotWidth;
    const mapY = (y) => marginTop + ((maxY - y) / (maxY - minY)) * plotHeight;

    // Draw Axes
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(marginLeft, marginTop);
    ctx.lineTo(marginLeft, canvas.height - marginBottom);
    ctx.lineTo(canvas.width - marginRight, canvas.height - marginBottom);
    ctx.stroke();
    
    // Draw Ticks
    ctx.fillStyle = 'black';
    ctx.font = '12px Arial';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    
    // Y Ticks
    for (let val = Math.ceil(minY); val <= Math.floor(maxY); val += 0.5) {
        const y = mapY(val);
        ctx.fillText(val.toFixed(1), marginLeft - 10, y);
        ctx.beginPath();
        ctx.moveTo(marginLeft - 5, y);
        ctx.lineTo(marginLeft, y);
        ctx.stroke();
    }
    
    // X Ticks - dynamic based on view
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xTickStep = 0.05;
    const startX = Math.ceil(viewMinX / xTickStep) * xTickStep;
    for (let val = startX; val <= viewMaxX; val += xTickStep) {
        const x = mapX(val);
        if (x > marginLeft && x < canvas.width - marginRight) {
            ctx.fillText(val.toFixed(2), x, canvas.height - marginBottom + 10);
            ctx.beginPath();
            ctx.moveTo(x, canvas.height - marginBottom);
            ctx.lineTo(x, canvas.height - marginBottom + 5);
            ctx.stroke();
        }
    }

    // Plot waves
    for (let i = 0; i < testCount; i++) {
        const color = `hsl(${(i * 360) / testCount}, 70%, 70%)`; // Lighter colors for lines
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        // Plot points in view
        let firstPoint = true;
        for (let j = startIndex; j < 49; j++) {
            const p = testData[i][j];
            const x = mapX(p.x);
            const y = mapY(p.y);
            if (firstPoint) {
                ctx.moveTo(x, y);
                firstPoint = false;
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
        
        // Actual Target (Red)
        const target = testData[i][49];
        const tx = mapX(target.x);
        const ty = mapY(target.y);
        
        ctx.fillStyle = 'red';
        ctx.beginPath();
        ctx.arc(tx, ty, 6, 0, 2 * Math.PI);
        ctx.fill();
        
        // Predicted Target (Green)
        const predY = predValues[i];
        const px = mapX(target.x); // Same x as target
        const py = mapY(predY);
        
        ctx.fillStyle = 'green';
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, 2 * Math.PI);
        ctx.fill();
        
        // Connect actual and predicted with a thin line to show error
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(tx, ty);
        ctx.lineTo(px, py);
        ctx.stroke();
        ctx.setLineDash([]);
    }
    
    statusDiv.innerText = "Training and Evaluation complete!";
}

function drawLossGraph() {
    const canvas = document.getElementById('lossCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (lossHistory.length === 0) return;
    
    const padding = 40;
    const width = canvas.width - 2 * padding;
    const height = canvas.height - 2 * padding;
    
    // Find max loss for scaling
    let maxLoss = 0;
    lossHistory.forEach(h => {
        maxLoss = Math.max(maxLoss, h.loss);
        if (h.val_loss) maxLoss = Math.max(maxLoss, h.val_loss);
    });
    
    // Draw axes
    ctx.beginPath();
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 1;
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, canvas.height - padding);
    ctx.lineTo(canvas.width - padding, canvas.height - padding);
    ctx.stroke();
    
    // Draw labels
    ctx.fillStyle = 'black';
    ctx.textAlign = 'center';
    ctx.font = '12px Arial';
    ctx.fillText("Epochs", canvas.width / 2, canvas.height - 10);
    
    ctx.save();
    ctx.translate(15, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("MSE Loss", 0, 0);
    ctx.restore();
    
    // Plot training loss (Orange)
    ctx.beginPath();
    ctx.strokeStyle = 'orange';
    ctx.lineWidth = 2;
    
    for (let i = 0; i < lossHistory.length; i++) {
        const x = padding + (i / (lossHistory.length - 1 || 1)) * width;
        const y = canvas.height - padding - (lossHistory[i].loss / maxLoss) * height;
        
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Plot validation loss (Blue)
    ctx.beginPath();
    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 2;
    
    for (let i = 0; i < lossHistory.length; i++) {
        if (lossHistory[i].val_loss === undefined) continue;
        const x = padding + (i / (lossHistory.length - 1 || 1)) * width;
        const y = canvas.height - padding - (lossHistory[i].val_loss / maxLoss) * height;
        
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
    
    // Legend
    ctx.textAlign = 'right';
    ctx.fillStyle = 'orange';
    ctx.fillText("Train", canvas.width - padding, padding);
    ctx.fillStyle = 'blue';
    ctx.fillText("Val", canvas.width - padding, padding + 15);

    // Draw max loss value
    ctx.fillStyle = 'black';
    ctx.textAlign = 'right';
    ctx.fillText(maxLoss.toFixed(4), padding - 5, padding);
    ctx.fillText("0", padding - 5, canvas.height - padding);

    // Draw X ticks (Epochs)
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const numXTicks = Math.min(lossHistory.length, 5);
    for (let i = 0; i < numXTicks; i++) {
        const index = Math.round((i / (numXTicks - 1 || 1)) * (lossHistory.length - 1));
        const item = lossHistory[index];
        const x = padding + (index / (lossHistory.length - 1 || 1)) * width;
        
        ctx.beginPath();
        ctx.moveTo(x, canvas.height - padding);
        ctx.lineTo(x, canvas.height - padding + 5);
        ctx.stroke();
        
        ctx.fillText(item.epoch, x, canvas.height - padding + 5);
    }
}

function generateSinWaves(count, minAmp, maxAmp, minWave, maxWave, negProb = 50) {
    const newData = [];
    for (let i = 0; i < count; i++) {
        const amp = minAmp + Math.random() * (maxAmp - minAmp);
        const wave = minWave + Math.random() * (maxWave - minWave);
        const isNegative = Math.random() * 100 < negProb;
        const sign = isNegative ? -1 : 1;
        
        const points = [];
        for (let j = 0; j < 50; j++) {
            const x = j / 49; // 0 to 1, 50 points
            const y = sign * amp * Math.sin(2 * Math.PI * x / wave);
            points.push({x, y});
        }
        newData.push(points);
    }
    return newData;
}

function generateWaves() {
    const numPoints = parseInt(document.getElementById('numPoints').value);
    const minAmp = parseFloat(document.getElementById('minAmp').value);
    const maxAmp = parseFloat(document.getElementById('maxAmp').value);
    const minWave = parseFloat(document.getElementById('minWave').value);
    const maxWave = parseFloat(document.getElementById('maxWave').value);
    const negProb = parseFloat(document.getElementById('negProb').value);

    generatedWaves = generateSinWaves(numPoints, minAmp, maxAmp, minWave, maxWave, negProb);
}

function plotWaves() {
    const startIndex = parseInt(document.getElementById('startIndex').value);
    const numToShow = parseInt(document.getElementById('numToShow').value);
    const canvas = document.getElementById('waveCanvas');
    const legend = document.getElementById('legend');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    legend.innerHTML = '';

    if (generatedWaves.length === 0) {
        return;
    }

    // Find global min and max y for scaling
    let minY = Infinity;
    let maxY = -Infinity;
    for (let i = startIndex; i < startIndex + numToShow && i < generatedWaves.length; i++) {
        generatedWaves[i].forEach(p => {
            minY = Math.min(minY, p.y);
            maxY = Math.max(maxY, p.y);
        });
    }

    // Add some padding to Y range so waves don't touch top/bottom exactly
    const yRange = maxY - minY;
    if (yRange === 0) { // Handle flat line case
        minY -= 1;
        maxY += 1;
    } else {
        minY -= yRange * 0.1;
        maxY += yRange * 0.1;
    }

    const marginTop = 20;
    const marginBottom = 40;
    const marginLeft = 60;
    const marginRight = 100; // Extra space for target labels

    const plotWidth = canvas.width - marginLeft - marginRight;
    const plotHeight = canvas.height - marginTop - marginBottom;

    // Helper to map data coordinates to canvas coordinates
    const mapX = (x) => marginLeft + x * plotWidth;
    const mapY = (y) => marginTop + ((maxY - y) / (maxY - minY)) * plotHeight;

    // Draw Axes
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 1;
    ctx.beginPath();
    // Y Axis
    ctx.moveTo(marginLeft, marginTop);
    ctx.lineTo(marginLeft, canvas.height - marginBottom);
    // X Axis
    ctx.lineTo(canvas.width - marginRight, canvas.height - marginBottom);
    ctx.stroke();

    // Draw Ticks and Labels
    ctx.fillStyle = 'black';
    ctx.font = '12px Arial';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    // Y Ticks: 0 and every 0.5
    let startY = Math.ceil(minY / 0.5) * 0.5;
    // Fix floating point precision issues by using epsilon
    const epsilon = 0.0001;
    for (let val = startY; val <= maxY + epsilon; val += 0.5) {
        const y = mapY(val);
        ctx.beginPath();
        ctx.moveTo(marginLeft - 5, y);
        ctx.lineTo(marginLeft, y);
        ctx.stroke();
        ctx.fillText(val.toFixed(1), marginLeft - 10, y);
    }

    // X Ticks: every 0.25
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let val = 0; val <= 1.0 + epsilon; val += 0.25) {
        const x = mapX(val);
        ctx.beginPath();
        ctx.moveTo(x, canvas.height - marginBottom);
        ctx.lineTo(x, canvas.height - marginBottom + 5);
        ctx.stroke();
        ctx.fillText(val.toFixed(2), x, canvas.height - marginBottom + 10);
    }

    for (let i = startIndex; i < startIndex + numToShow && i < generatedWaves.length; i++) {
        const waveIndex = i - startIndex; // relative index for color
        const color = `hsl(${(waveIndex * 360) / numToShow}, 100%, 50%)`; // Different colors
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        // Plot first 49 points as line
        for (let j = 0; j < 49; j++) {
            const p = generatedWaves[i][j];
            const x = mapX(p.x);
            const y = mapY(p.y);
            if (j === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();

        // Plot target (50th point) as red dot
        const target = generatedWaves[i][49];
        const tx = mapX(target.x);
        const ty = mapY(target.y);
        ctx.fillStyle = 'red';
        ctx.beginPath();
        ctx.arc(tx, ty, 5, 0, 2 * Math.PI);
        ctx.fill();

        // Label the target
        ctx.fillStyle = 'black';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.font = '12px Arial';
        ctx.fillText(`target${i}`, tx + 10, ty);

        // Add to Legend
        const legendItem = document.createElement('div');
        legendItem.className = 'legend-item';
        legendItem.innerHTML = `<span class="color-box" style="background-color: ${color};"></span> Wave ${i}`;
        legend.appendChild(legendItem);
    }
}