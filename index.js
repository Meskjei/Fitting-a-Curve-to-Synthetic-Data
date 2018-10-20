function generateData(numPoints, coeff, sigma = 0.04) {
  return tf.tidy(() => {
    const [a, b, c, d] = [
      tf.scalar(coeff.a), tf.scalar(coeff.b), tf.scalar(coeff.c),
      tf.scalar(coeff.d)
    ];

    const xs = tf.randomUniform([numPoints], -1, 1);

    // Generate polynomial data
    const three = tf.scalar(3, 'int32');
    const ys = a.mul(xs.pow(three))
      .add(b.mul(xs.square()))
      .add(c.mul(xs))
      .add(d)
      // Add random noise to the generated data
      // to make the problem a bit more interesting
      .add(tf.randomNormal([numPoints], 0, sigma));

    // Normalize the y values to the range 0 to 1.
    const ymin = ys.min();
    const ymax = ys.max();
    const yrange = ymax.sub(ymin);
    const ysNormalized = ys.sub(ymin).div(yrange);
    ysNormalized.print(); 
    return {
      xs, 
      ys: ysNormalized
    };
  })
}
//初始化方程系数
let a = tf.variable(tf.scalar(Math.random()));
console.log('预测前a的值为：');
a.print();
let b = tf.variable(tf.scalar(Math.random()));
console.log('预测前b的值为：');
b.print();
let c = tf.variable(tf.scalar(Math.random()));
console.log('预测前c的值为：');
b.print();
let d = tf.variable(tf.scalar(Math.random()));
console.log('预测前d的值为：');
d.print();
//确定模型
function predict(x){
   return tf.tidy(() => {
      return a.mul(x.pow(tf.scalar(3)))
        .add(b.mul(x.square()))
        .add(c.mul(x))
        .add(d);
   });
}
//确定损失方程
function loss(prediction, labels){
  let meanSquareError =  prediction.sub(labels).square().mean();
  return meanSquareError;
}

//确定优化器
let learningRate = 0.2;
let optimizer = tf.train.sgd(learningRate);

//定义训练循环
function train(xs, ys, numIterations){

  for (let iter = 0; iter < numIterations; iter++) {
    optimizer.minimize(() => {
      const predsYs = predict(xs);
      return loss(predsYs, ys);
    });
  }
}

async function learnCoefficients(){
  let trueCoefficients = {a: -.8, b: -.2, c: .9, d: .5};
  let trainingData = generateData(100, trueCoefficients);
  const predictionsBefore = predict(trainingData.xs);
  await train(trainingData.xs, trainingData.ys, 1000);
  const predictionsAfter = predict(trainingData.xs);
  a.print();
  b.print();
  c.print();
  d.print();
  predictionsBefore.dispose();
  predictionsAfter.dispose(); 
}



learnCoefficients();
