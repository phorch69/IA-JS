import * as tf from '@tensorflow/tfjs';

// Creamos un modelo secuencial simple con una capa densa.
const model = tf.sequential();

// Capa densa con 1 unidad, que toma una sola entrada (el número que ingrese el usuario).
model.add(tf.layers.dense({ units: 1, inputShape: [1], activation: 'linear' }));

// Compilamos el modelo con un optimizador y una función de pérdida.
model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

// Datos de entrenamiento (usaremos un simple modelo Y = 2 * X)
const xs = tf.tensor([1, 2, 3, 4, 5], [5, 1]);
const ys = tf.tensor([2, 4, 6, 8, 10], [5, 1]);

// Entrenamos el modelo con los datos de entrada y salida.
async function trainModel() {
  await model.fit(xs, ys, { epochs: 100 });
  console.log("Modelo entrenado!");
}

// Función para realizar la predicción con el modelo.
function predict(x) {
  const input = tf.tensor([x], [1, 1]);
  const result = model.predict(input);
  return result.dataSync()[0];
}

// Llamamos a la función de entrenamiento.
trainModel();

// Función que se llama cuando el usuario hace clic en el botón "Predecir"
document.getElementById('predictButton').addEventListener('click', () => {
  const inputValue = parseFloat(document.getElementById('inputValue').value);
  if (isNaN(inputValue)) {
    alert('Por favor, ingresa un número válido');
    return;
  }

  const prediction = predict(inputValue);
  document.getElementById('predictionResult').textContent = prediction.toFixed(2);
});
