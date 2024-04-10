use ndarray::{
    s, Array, Array1, Array2, Array3, ArrayBase, ArrayD, Axis, Data, Dimension, Ix2, RemoveAxis,
    Zip,
};
//use ndarray::Array;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::error::Error;

struct ConvLayer {
    kernel: Array2<f32>,
    stride: usize,
    padding: usize,
}

impl ConvLayer {
    fn new(kernel: Array2<f32>, stride: usize, padding: usize) -> ConvLayer {
        ConvLayer {
            kernel,
            stride,
            padding,
        }
    }

    fn forward(&self, input: Array3<f32>) -> Array3<f32> {
        println!("Kernel shape: {:?}", self.kernel.dim());
        println!("Stride: {}", self.stride);
        println!("Padding: {}", self.padding);
        println!("input: {:?}", input.dim());
        let kernel_size = self.kernel.dim();
        let (pad_height, pad_width) = (self.padding, self.padding);
        let (stride_y, stride_x) = (self.stride, self.stride);

        //calculate output dims
        let output_height = ((input.dim().1 as isize - kernel_size.0 as isize
            + 2 * pad_height as isize)
            / stride_y as isize
            + 1) as usize;
        let output_width = ((input.dim().2 as isize - kernel_size.1 as isize
            + 2 * pad_width as isize)
            / stride_x as isize
            + 1) as usize;

        // Initialize the output with zeros
        let mut output = Array3::zeros((1, output_height, output_width));
        // Apply padding to the input if needed
        let padded_input = if pad_height > 0 || pad_width > 0 {
            let mut padded = Array3::zeros((
                input.dim().0,
                input.dim().1 + 2 * pad_height,
                input.dim().2 + 2 * pad_width,
            ));
            println!("Padded shape before assignment: {:?}", padded.dim());
            let (ph, pw) = (pad_height, pad_width);
            println!("ph:{}, pw:{}", ph, pw);
            //padded
            //  .slice_mut(s![0, ph..ph + input.dim().1, pw..pw + input.dim().2])
            // .assign(&input);

            println!("Shape of input: {:?}", input.dim());
            let slice_dims = s![.., ph..ph + input.dim().1, pw..pw + input.dim().2];
            padded.slice_mut(&slice_dims).assign(&input);
            padded
        } else {
            input.clone()
        };
        // println!("padded input new: {:?}", padded_input);

        // Perform the convolution operation
        for y in 0..output_height {
            for x in 0..output_width {
                let y_start = y * stride_y;
                let x_start = x * stride_x;
                let input_section = padded_input.slice(s![
                    0,
                    y_start..y_start + kernel_size.0,
                    x_start..x_start + kernel_size.1
                ]);

                // Element-wise multiplication and summation
                let conv_sum = (&self.kernel * &input_section).sum();
                //println!("output from conv:{:?}", output);
                output[[0, y, x]] = conv_sum;
            }
        }

        output
    }
}

enum ActivationFunction {
    Relu,
    Sigmoid,
    Softmax,
}

impl ActivationFunction {
    fn apply<D>(
        &self,
        input: &ArrayBase<impl Data<Elem = f32>, D>,
    ) -> Result<ArrayD<f32>, Box<dyn Error>>
    where
        D: Dimension + RemoveAxis,
    {
        match self {
            ActivationFunction::Relu => Ok(input.mapv(|a| a.max(0.0)).into_dyn()),
            ActivationFunction::Sigmoid => Ok(input.mapv(|a| 1.0 / (1.0 + (-a).exp())).into_dyn()),
            ActivationFunction::Softmax => self.softmax(input),
        }
    }
    fn softmax<D>(
        &self,
        input: &ArrayBase<impl Data<Elem = f32>, D>,
    ) -> Result<ArrayD<f32>, Box<dyn Error>>
    where
        D: Dimension + ndarray::RemoveAxis,
    {
        // Determine if the input is 1D or 2D by checking the number of dimensions.
        match input.ndim() {
            1 => {
                //For 1D input, process as a single vector (row).
                let max_logit = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exps = input.mapv(|x| (x - max_logit).exp());
                let sum_exps = exps.sum();
                Ok(exps.mapv(|x| x / sum_exps).into_dyn())
            }
            2 => {
                let max_logit = input.map_axis(Axis(1), |row| {
                    *row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
                });
                let max_logit_broadcasted = max_logit; //.insert_axis(Axis(0));
                println!("max_logit_broadcasted: {:?}", max_logit_broadcasted.dim());
                let exps = (input - &max_logit_broadcasted).mapv(f32::exp);
                let sum_exps = exps.sum_axis(Axis(1));
                // Avoid direct division that leads to broadcasting failure.
                Ok(exps / &sum_exps.into_dyn())
            }
            _ => Err("Softmax only supports 2d and 1d arrays!".into()),
        }
        // Ensure the output is dynamically dimensioned for flexibility
    }
}

struct FullyConnected {
    weights: Array2<f32>,
    bias: Array1<f32>,
}

impl FullyConnected {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        let weights = Array2::random((output_dim, input_dim), Uniform::new(0.0, 1.0));
        //Array2::<f32>::zeros((output_dim, input_dim)); // Simplified initialization
        let bias = Array1::random(output_dim, Uniform::new(0.0, 1.0));
        //Array1::<f32>::zeros(output_dim);

        FullyConnected { weights, bias }
    }

    fn forward(&self, input: &ArrayD<f32>) -> Array2<f32> {
        let input_2d = input
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("Input must be 2D for dot product");
        let mut output = self.weights.dot(&input_2d);
        println!("Output Shape: {:?}", output.shape());
        let bias_reshaped = self.bias.view().insert_axis(Axis(1)); //need to reshape to match the
                                                                   //dimensions of the output array
        output += &bias_reshaped;

        println!("Result: {:?}", output);
        output
    }

    fn backward(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut output = self.weights.dot(input);
        let bias_reshaped = self.bias.view().insert_axis(Axis(1));
        output
    }
}

fn flatten(input: &ArrayD<f32>) -> ArrayD<f32> {
    let flattened_len = input.len(); // Total number of elements in the input
    let flattened = input.clone().into_shape((1, flattened_len)).unwrap();
    flattened.into_dyn()
}

enum LossFunction {
    MeanSquaredError,
    CrossEntropy,
    // You can add more loss functions here later, e.g., CrossEntropy
}

impl LossFunction {
    pub fn compute_loss(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        match self {
            LossFunction::MeanSquaredError => self.mse(predictions, targets),
            LossFunction::CrossEntropy => self.cross_entropy_loss(predictions, targets),
        }
    }

    fn mse(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        let diff = predictions - targets;
        let mse = diff.mapv(|a| a.powi(2)).mean().unwrap_or(0.0);
        mse
    }

    fn cross_entropy_loss(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        // Ensure predictions are clipped to avoid log(0)
        let clipped_predictions = predictions.mapv(|p| p.max(1e-7).min(1.0 - 1e-7));
        let loss = -targets * clipped_predictions.mapv(|p| p.ln());
        loss.sum() / predictions.shape()[0] as f32
    }
    pub fn compute_gradient(
        &self,
        predictions: &Array2<f32>,
        targets: &Array2<f32>,
    ) -> Array2<f32> {
        match self {
            LossFunction::MeanSquaredError => self.mse_gradient(predictions, targets),
            LossFunction::CrossEntropy => self.cross_entropy_gradient(predictions, targets),
        }
    }

    fn mse_gradient(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Array2<f32> {
        let gradient = 2.0 * (predictions - targets) / predictions.shape()[0] as f32;
        gradient
    }

    fn cross_entropy_gradient(
        &self,
        predictions: &Array2<f32>,
        targets: &Array2<f32>,
    ) -> Array2<f32> {
        let gradient = predictions - targets;
        gradient / predictions.shape()[0] as f32 // Normalize by batch size if necessary
    }
}

fn main() {
    println!("Hello, Lets get started!");
    let kernel = Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0])
        .expect("Failed to create kernel");

    let shape = (1, 28, 28);
    // Create a ConvLayer instance with the specified kernel, stride, and padding
    let conv_layer = ConvLayer::new(kernel, 1, 1);
    let input = Array3::random(shape, Uniform::new(0.0, 1.0));
    println!("Shape of input array: {:?}", input.shape());

    let output = conv_layer.forward(input);
    let output_activation = ActivationFunction::Relu.apply(&output);
    let flattened_output = flatten(&output_activation.unwrap());
    let input_features = flattened_output.len_of(Axis(1));
    let fc = FullyConnected::new(input_features, 10);
    let fc_input = flattened_output.t().to_owned();
    let fc_output = fc.forward(&fc_input);
    let prediction = ActivationFunction::Softmax.apply(&fc_output);

    println!("{:?}", prediction);
}
