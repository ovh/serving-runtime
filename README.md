# Serving Runtime
Exposes a serialized machine learning model through a HTTP API written in Java.

![TUs & TIs](https://github.com/ovh/serving-runtime/workflows/TUs%20&%20TIs/badge.svg?branch=master) [![Maintenance](https://img.shields.io/maintenance/yes/2020.svg)]() [![Chat on gitter](https://img.shields.io/gitter/room/ovh/ai.svg)](https://gitter.im/ovh/ai) 
 
 **This project is under active development**
 
## Description

The purpose of this project is to expose a generic HTTP API from a machine learning serialized models.

Supported serialized models are :
* [ONNX][ONNX] `1.5`
* TensorFlow `<=1.15` SavedModel or HDF5
 
## Prerequisites

* Maven for compiling the project
* `Java 11` for running the project

`HDF5` serialization format is supported through a conversion into `SavedModel` format. That conversion relies on following dependencies :

* Python `3.7`
* TensorFlow `<=1.15` (`pip install tensorflow`)

### HDF5 support (Optional)
<aside class="notice">
<p>If you use the API from the docker image this step is not necessary as it will be built within the image.</p>
</aside>

The Tensorflow module requires the support of HDF5 files through the creation of an executable `h5_converter` wich exports the model from HDF5 file to a Tensorflow SavedModel (`.pb`).  

To generate the converter simply use the initialize goal of the `Makefile`:
```bash
make initialize
```
The generated executable can be found here: `evaluator-tensorflow/h5_converter/dist/h5_converter`

## Build & Launch the project locally
Several profiles are available depending on the support you require for the built project.
- `full` which includes both Tensorflow and ONNX, requires the [ONNX support](#onnx-support-optional) and [HDF5 support](#hdf5-support-optional)
- `tensorflow` which only includes Tensorflow, requires the [HDF5 support](#hdf5-support-optional)
- `onnx` which only includes ONNX, requires the [ONNX support](#onnx-support-optional). 

Set your desired profile:
```bash
export MAVEN_PROFILE=<your-profile>
```
If not specified the default profile is set to `full`.

### Launch tests

```bash
make test MAVEN_PROFILE=$MAVEN_PROFILE
```

### Building JAR

```bash
make build MAVEN_PROFILE=$MAVEN_PROFILE
```

The JAR could then be found in `api/target/api-*.jar`

### Launching JAR

In the following command, replace `<jar-path>` with the path on your compiled jar and `<model-path>` with the directory where to find your serialized model.

```bash
java -Dfiles.path=<model-path> -jar <jar-path>
```

If you wish to load a model from a HDF5 model you will need to specify the path to the executable generated in [HDF5 support](#hdf5-support-optional).

```bash
java -Dfiles.path=<model-path> -Devaluator.tensorflow.h5_converter.path=<path-to-h5-converter> -jar <jar-path>
```

Inside the `<model-path>` it will look for the first file ending with :
* `.onnx` for an ONNX model
* `.pb` for a TensorFlow SavedModel
* `.h5` for a HDF5 model

#### Available parameters

On the launch command you can also specify the following parameters :

* `-Dserver.port` : the host port to request for the http server
* `-Dswagger.title` : The title that will be dispayed on the swagger
* `-Dswagger.description` : The description that will be displayed on the swagger

## Build & Launch the project using docker

### Building the docker container

```bash
make docker-build-api MAVEN_PROFILE=$MAVEN_PROFILE
```

It will build the docker image `serving-runtime-$MAVEN_PROFILE:latest`

### Running the docker container

In the following command, replace `<model-path>` with the absolute path on directory where to find your serialized model.

```bash
docker run --rm -it -p 8080:8080 -v <model-path>:/deployments/models serving-runtime-$MAVEN_PROFILE:latest
```

## Using the API

By default the API will be running on `http://localhost:8080`. Reaching this URL in your browser will display the SwaggerUI describing the API for your model.

There is 2 routes available in each models :

* `/describe` : Describe your model (what are the inputs, outputs and transformations)
* `/eval` : Send expected inputs on model and receive expected outputs results

### Describe the models inputs and outputs

Each serialized model takes a list of named tensors as **inputs** and also returns a list of named tensors as **outputs**. 

A **named tensors** is a **N-Dimensional array** with :

* A identifier name. Example: `my-tensor-name`
* A data type. Example: `integer` or `double` or `string`
* A shape. Example: `(5)` for a vector of length **5**, `(3, 2)` for a matrix which first dimension is of size **3** and second dimension is of size **2**. Etc.

You can get access to the model inputs and outputs by calling the http `GET` method on `/describe` path of the model.

#### Example of a describe query with curl

```bash
curl \
    -X GET \
    http://<your-model-url>/describe
```

#### Example of a describe response

You will get a **JSON** object describing the list of **inputs tensors** that are needed to query your model as well as the list of **outputs tensors** that will be returning.

```json
{
    "inputs": [
        {
            "name": "sepal_length",
            "type": "float",
            "shape": [-1]
        },
        {
            "name": "sepal_width",
            "type": "float",
            "shape": [-1]
        },
        {
            "name": "petal_length",
            "type": "float",
            "shape": [-1]
        },
        {
            "name": "petal_width",
            "type": "float",
            "shape": [-1]
        }
    ],
    "outputs": [
        {
            "name": "output_label",
            "type": "long",
            "shape": [-1]
        },
        {
            "name": "output_probability",
            "type": "float",
            "shape": [-1, 2]
        }
    ]
}
```

In this example, the deployed model is waiting for 4 tensors as inputs :

* `sepal_length` of shape `(-1)` (i.e. a vector of any size)
* `sepal_width` of shape `(-1)` (i.e. a vector of any size)
* `petal_length` of shape `(-1)` (i.e. a vector of any size)
* `petal_width` of shape `(-1)` (i.e. a vector of any size)

It will answer a response with 2 tensors as outputs :

* `output_label` of shape `(-1)` (i.e. a vector of any size)
* `output_probability` of shape `(-1, 2)` (i.e. a matrix which first dimension is of any size and which second dimension is of size 2)

### Query the model

Once you know what kind of **input tensors** are needed by the model, just fill a correct **body** on your **HTTP query** with your wanted representation of tensor (see below) and send it to the model with a `POST` method on the path `/eval`.

Two attached headers are available for your query:

* The [Content-Type][Content Type Header] header indicating the [media type][Media Type] of your input tensors data contained in your body message.
* The (optional) [Accept][Accept Header] header indicating what kind of [media type][Media Type] your want to receive for output tensors in the response body. The default `Accept` header if you don't provide one will be `application/json`.

### Supported Content-Type headers

* `application/json` : A json document which **key** are the **input tensors** names and **values** are the n-dimensional json arrays matching your tensors.

* `image/png` : A bytes content which representation is a **png** encoded image.
* `image/jpeg` : A bytes content which representation is a **jpeg** encoded image.

>
> `image/png` and `image/jpeg` are only available for models taking a single tensor as input. That tensor's shape should also be compatible with an image representation.
>

* `multipart/form-data` : A multipart body, each part of which is named by an **input tensor**.

> 
> Each part (i.e. tensor) in the **multipart** should have its own **Content-Type**
> 

### Supported Accept headers

* `application/json` : A JSON document which **key** is the **output tensors** names and **values** are the n-dimensional json arrays matching your tensors.

* `image/png` : A bytes content which representation is a **png** encoded image.
* `image/jpeg` : A bytes content which representation is a **jpeg** encoded image.

>
> `image/png` and `image/jpeg` are only available for models returning a single tensor as output. That tensor's shape should also be compatible with an image representation.
>

* `text/html` : A HTML document displaying the **output tensors** representation.
* `multipart/form-data` : A multipart body, each part of which is named by an **output tensor** and the content is the tensor json representation.

>
> If you want some of the output tensors in `multipart/form-data` and `text/html` header to be interpreted as an image, you can specify it as a parameter in the header.
>
> **Example** : The header `text/html; tensor_1=image/png; tensor_2=image/png` returns the global response as HTML content. Inside the HTML page, `tensor_1` and `tensor_2` are displayed as **png** images.
>

### Tensor interpretable as image

For a tensor to be interpretable as image raw data, it should be of a compatible shape in your exported model. Here are the supported ones :

* `(x, y, z, 1)` : Batch of **x** grayscale images with **y** pixels height and **z** pixels width 
* `(x, 1, y, z)` : Batch of **x** grayscale images with **y** pixels height and **z** pixels width
* `(x, y, z, 3)` : Batch of **x** RGB images with **y** pixels height and **z** pixels width. The last dimension should be the array of `(red, green, blue)` components.
* `(x, 3, y, z)` : Batch of **x** RGB images with **y** pixels height and **z** pixels width. The last dimension should be the array of `(red, green, blue)` components.
* `(y, z, 1)` : Single grayscale image with **y** pixels height and **z** pixels width
* `(1, y, z)` : Single grayscale image with **y** pixels height and **z** pixels width
* `(y, z, 3)` : Single RGB image with **y** pixels height and **z** pixels width. The last dimension should be the array of `(red, green, blue)` components.
* `(3, y, z)` : Single RGB image with **y** pixels height and **z** pixels width. The last dimension should be the array of `(red, green, blue)` components.

## Examples

### Example of a query with curl for a single prediction

In the following example, we want to receive a prediction from our model for the following item :

* `sepal_length` : 0.1
* `sepal_width` : 0.2
* `petal_length` : 0.3
* `petal_width` : 0.4

```bash
curl \
    -H 'Content-Type: application/json' \
    -H 'Accept: application/json' \
    -X POST \
    -d '{
        "stepal_length": 0.1,
        "stepal_width": 0.2,
        "petal_length": 0.3,
        "petal_width": 0.4
    }' \
    http://<your-model-url>/eval
```

### Example of response for a single prediction


* HTTP Status code: `200`
* Header: `Content-Type: application/json`

```json
{
    "output_label": 0,
    "output_probability": [0.88, 0.12]
}
```

In this example, our model predicts the **output_label** for our **input item** to be `0` with the following probabilities :

* 88% of chance to be `0`
* 12% of chance to be `1`

### Example of query with curl for several predictions in one call

In the following example, we want to receive a prediction from our model for the two following items :

**First Item**

* `sepal_length` : 0.1
* `sepal_width` : 0.2
* `petal_length` : 0.3
* `petal_width` : 0.4

**Second Item**

* `sepal_length` : 0.2
* `sepal_width` : 0.3
* `petal_length` : 0.4
* `petal_width` : 0.5

**Query**

```bash
curl \
    -H 'Content-Type: application/json' \
    -H 'Accept: application/json' \
    -X POST \
    -d '{
        "stepal_length": [0.1, 0.2],
        "stepal_width": [0.2, 0.3],
        "petal_length": [0.3, 0.4],
        "petal_width": [0.4, 0.5]
    }' \
    http://<your-model-url>/eval
```

### Example of response for several predictions in one call

* HTTP Status code: `200`
* Header: `Content-Type: application/json`

```json
{
    "output_label": [0, 1],
    "output_probability": [
        [0.88, 0.12],
        [0.01, 0.99]
    ]
}
```

In this example, our model predicts the **output_label** for our **first input item** to be `0` with the following probabilities :

* 88% of chance to be `0`
* 12% of chance to be `1`

It also predicts the **output_label** for our **second input item** to be `1` with the following probabilities :

* 1% of chance to be `0`
* 99% of chance to be `1`

# Related links
 
 * Contribute: https://github.com/ovh/serving-runtime/blob/master/CONTRIBUTING.md
 * Report bugs: https://github.com/ovh/serving-runtime/issues
 
# License
 
See https://github.com/ovh/serving-runtime/blob/master/LICENSE

[Content Type Header]: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Type
[Accept Header]: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Accept
[Media Type]: https://developer.mozilla.org/en-US/docs/Glossary/MIME_type
[ONNX]: https://onnx.ai/