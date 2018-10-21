const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));


function predict(x) {
  return tf.tidy(() => {
    return a.mul(x.pow(tf.scalar(2, 'int32'))) 
      .add(b.mul(x))
      .add(c);
  });
}

function loss(prediction, labels) {
  const squareMeanError = prediction.sub(labels).square().mean();
  return squareMeanError;
}

const numIterations = 360;
const learningRate = 0.02;
const optimizer = tf.train.sgd(learningRate);

async function train(xs, ys, numIterations) {
  for (let iter = 0; iter < numIterations; iter++) {
    optimizer.minimize(() => {
      const pred = predict(xs);
      return loss(pred, ys);
    });
    await tf.nextFrame();
  }
}

function generateData(numPoints, coeff, sigma = 0.5) {
  return tf.tidy(() => {
    const [a, b, c] = [
      tf.scalar(coeff.a),
      tf.scalar(coeff.b),
      tf.scalar(coeff.c)
    ];
  
    const xs = tf.randomUniform([numPoints], -4, 2);
    const ys = a.mul(xs.pow(tf.scalar(2, 'int32')))
      .add(b.mul(xs))
      .add(c)
      .add(tf.randomNormal([numPoints], 0, sigma));

    return {
      xs,
      ys: ys
    };
  })
}

async function plotData(xs, ys, preds) {
  const xvals = await xs.data();
  const yvals = await ys.data();
  const predVals = await preds.data();
  
  const valuesBefore = Array.from(xvals).map((x, i) => {
    return [xvals[i], yvals[i]];
  });
  const valuesAfter= Array.from(xvals).map((x, i) => {
    return [xvals[i], predVals[i]];
  });
  console.log(valuesBefore);
  // 二维数组排序
  valuesAfter.sort(function(x, y) {
    return x[0] - y[0];
  });
  curveChart.setOption({
    xAxis: {
      min: -4,
      max: 2
    },
    yAxis: {
      min: 0,
      max: 10
    },
    series: [{
      symbolSize: 12,
      data: valuesBefore,
      type: 'scatter'
    },{
      data: valuesAfter,
      encode: {
        x: 0,
        y: 1
      },
      type: 'line'
    }]
  });
}

async function learnCoefficients() {
  const trueCoefficients = {a: 1.0, b: 2.0, c: 1.0};
  // 生成有误差的训练数据
  const trainingData = generateData(200, trueCoefficients);
  // 训练模型
  await train(trainingData.xs, trainingData.ys, numIterations);
  
  // 预测数据
  const predictionsAfter = predict(trainingData.xs);
  a.print();
  // 绘制散点图及拟合曲线
  await plotData(trainingData.xs, trainingData.ys, predictionsAfter);
  predictionsAfter.dispose();
}


const curveChart = echarts.init(document.getElementById('chart'));
learnCoefficients();