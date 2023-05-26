{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "235414e0",
   "metadata": {
    "papermill": {
     "duration": 0.00777,
     "end_time": "2023-05-26T14:48:03.966816",
     "exception": false,
     "start_time": "2023-05-26T14:48:03.959046",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**Deep Learning with R **"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fed006c9",
   "metadata": {
    "papermill": {
     "duration": 0.005984,
     "end_time": "2023-05-26T14:48:03.979634",
     "exception": false,
     "start_time": "2023-05-26T14:48:03.973650",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Setting up the environment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f669f415",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-26T14:48:03.996230Z",
     "iopub.status.busy": "2023-05-26T14:48:03.993942Z",
     "iopub.status.idle": "2023-05-26T14:48:06.192334Z",
     "shell.execute_reply": "2023-05-26T14:48:06.190623Z"
    },
    "papermill": {
     "duration": 2.209411,
     "end_time": "2023-05-26T14:48:06.195179",
     "exception": false,
     "start_time": "2023-05-26T14:48:03.985768",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "── \u001b[1mAttaching core tidyverse packages\u001b[22m ──────────────────────── tidyverse 2.0.0 ──\n",
      "\u001b[32m✔\u001b[39m \u001b[34mdplyr    \u001b[39m 1.1.2     \u001b[32m✔\u001b[39m \u001b[34mreadr    \u001b[39m 2.1.4\n",
      "\u001b[32m✔\u001b[39m \u001b[34mforcats  \u001b[39m 1.0.0     \u001b[32m✔\u001b[39m \u001b[34mstringr  \u001b[39m 1.5.0\n",
      "\u001b[32m✔\u001b[39m \u001b[34mggplot2  \u001b[39m 3.4.2     \u001b[32m✔\u001b[39m \u001b[34mtibble   \u001b[39m 3.2.1\n",
      "\u001b[32m✔\u001b[39m \u001b[34mlubridate\u001b[39m 1.9.2     \u001b[32m✔\u001b[39m \u001b[34mtidyr    \u001b[39m 1.3.0\n",
      "\u001b[32m✔\u001b[39m \u001b[34mpurrr    \u001b[39m 1.0.1     \n",
      "── \u001b[1mConflicts\u001b[22m ────────────────────────────────────────── tidyverse_conflicts() ──\n",
      "\u001b[31m✖\u001b[39m \u001b[34mdplyr\u001b[39m::\u001b[32mfilter()\u001b[39m masks \u001b[34mstats\u001b[39m::filter()\n",
      "\u001b[31m✖\u001b[39m \u001b[34mdplyr\u001b[39m::\u001b[32mlag()\u001b[39m    masks \u001b[34mstats\u001b[39m::lag()\n",
      "\u001b[36mℹ\u001b[39m Use the conflicted package (\u001b[3m\u001b[34m<http://conflicted.r-lib.org/>\u001b[39m\u001b[23m) to force all conflicts to become errors\n"
     ]
    }
   ],
   "source": [
    "library(tidyverse)\n",
    "library(keras)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "54b7c41b",
   "metadata": {
    "papermill": {
     "duration": 0.006177,
     "end_time": "2023-05-26T14:48:06.207788",
     "exception": false,
     "start_time": "2023-05-26T14:48:06.201611",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Checking the installed versions "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "137849c5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-26T14:48:06.257312Z",
     "iopub.status.busy": "2023-05-26T14:48:06.222721Z",
     "iopub.status.idle": "2023-05-26T14:48:10.699060Z",
     "shell.execute_reply": "2023-05-26T14:48:10.697003Z"
    },
    "papermill": {
     "duration": 4.488148,
     "end_time": "2023-05-26T14:48:10.702093",
     "exception": false,
     "start_time": "2023-05-26T14:48:06.213945",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table class=\"dataframe\">\n",
       "<caption>A data.frame: 2 × 2</caption>\n",
       "<thead>\n",
       "\t<tr><th></th><th scope=col>Package</th><th scope=col>Version</th></tr>\n",
       "\t<tr><th></th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th></tr>\n",
       "</thead>\n",
       "<tbody>\n",
       "\t<tr><th scope=row>keras</th><td>keras     </td><td>2.6.0 </td></tr>\n",
       "\t<tr><th scope=row>tensorflow</th><td>tensorflow</td><td>2.11.0</td></tr>\n",
       "</tbody>\n",
       "</table>\n"
      ],
      "text/latex": [
       "A data.frame: 2 × 2\n",
       "\\begin{tabular}{r|ll}\n",
       "  & Package & Version\\\\\n",
       "  & <chr> & <chr>\\\\\n",
       "\\hline\n",
       "\tkeras & keras      & 2.6.0 \\\\\n",
       "\ttensorflow & tensorflow & 2.11.0\\\\\n",
       "\\end{tabular}\n"
      ],
      "text/markdown": [
       "\n",
       "A data.frame: 2 × 2\n",
       "\n",
       "| <!--/--> | Package &lt;chr&gt; | Version &lt;chr&gt; |\n",
       "|---|---|---|\n",
       "| keras | keras      | 2.6.0  |\n",
       "| tensorflow | tensorflow | 2.11.0 |\n",
       "\n"
      ],
      "text/plain": [
       "           Package    Version\n",
       "keras      keras      2.6.0  \n",
       "tensorflow tensorflow 2.11.0 "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "installed.packages()[,c(1,3)] %>%\n",
    "  as.data.frame() %>%\n",
    "  filter(Package %in% c(\"keras\", \"tensorflow\"))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b2566347",
   "metadata": {
    "papermill": {
     "duration": 0.006376,
     "end_time": "2023-05-26T14:48:10.714962",
     "exception": false,
     "start_time": "2023-05-26T14:48:10.708586",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Dataset\n",
    "Our purpose is to classify images of handwritten digits. For this example,we make use of the MNIST dataset, a classic of the Machine Learning community.\n",
    "\n",
    "The dataset has the following characteristics:\n",
    "\n",
    "60,000 training images and 10,000 test images.\n",
    "Images of size 28 x 28 pixels.\n",
    "10 categories (digits from 0 to 9).\n",
    "Grayscale images: pixel values range between 0 (black) and 255 (white).\n",
    "Neural Networks require data in the shape of tensors. Tensors are algebraic objects with an arbitrary number of dimensions (D). For example, we can see vectors as 1D tensors and matrices as 2D tensors.\n",
    "\n",
    "In the case of images, we need a vector space able to convey:\n",
    "\n",
    "Number of images (N)\n",
    "Image height (H)\n",
    "Image width (W)\n",
    "Color channels (C), also known as color depth.\n",
    "Therefore, in Deep Learning tasks images are generally represented as 4D tensors with shape: N x H x W x C.\n",
    "\n",
    "In the case of grayscales images, the color channel is a single number (from 0 to 255) for each sample. Hence, it is possible to either omit the channel axis or leave it equal to one.\n",
    "\n",
    "Let us import the MNIST dataset from Keras and verify the shape of the training and test images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "4c85145d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-26T14:48:10.730236Z",
     "iopub.status.busy": "2023-05-26T14:48:10.728940Z",
     "iopub.status.idle": "2023-05-26T14:48:23.167366Z",
     "shell.execute_reply": "2023-05-26T14:48:23.165815Z"
    },
    "papermill": {
     "duration": 12.44909,
     "end_time": "2023-05-26T14:48:23.170317",
     "exception": false,
     "start_time": "2023-05-26T14:48:10.721227",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "X train shape:\t 60000 28 28\n",
      "X test shape:\t 10000 28 28\n",
      "Y train shape:\t 60000\n",
      "Y test shape:\t 10000"
     ]
    }
   ],
   "source": [
    "c(c(x_train, y_train), c(x_test, y_test)) %<-% keras::dataset_mnist()\n",
    "\n",
    "cat(\"\\nX train shape:\\t\", dim(x_train))\n",
    "cat(\"\\nX test shape:\\t\", dim(x_test))\n",
    "cat(\"\\nY train shape:\\t\", dim(y_train))\n",
    "cat(\"\\nY test shape:\\t\", dim(y_test))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4ff0dadb",
   "metadata": {
    "papermill": {
     "duration": 0.006753,
     "end_time": "2023-05-26T14:48:23.184176",
     "exception": false,
     "start_time": "2023-05-26T14:48:23.177423",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "The representation of input observations as tensors applies to any data type. For example, tabular data in the form of a csv file with 300 rows (samples) and 8 columns (features) can be seen as 2D tensors of shape 300 x 8.\n",
    "\n",
    "We can have a look at some samples by their corresponding label:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "370b6104",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-26T14:48:23.200161Z",
     "iopub.status.busy": "2023-05-26T14:48:23.198932Z",
     "iopub.status.idle": "2023-05-26T14:48:23.212200Z",
     "shell.execute_reply": "2023-05-26T14:48:23.210781Z"
    },
    "papermill": {
     "duration": 0.023613,
     "end_time": "2023-05-26T14:48:23.214512",
     "exception": false,
     "start_time": "2023-05-26T14:48:23.190899",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "plot_digits_by_class <- function(c) {\n",
    "  # plot first 15 digits of class c\n",
    "\n",
    "  # accepted labels are between 0 and 9\n",
    "  if (c > -1 & c < 10) {\n",
    "\n",
    "    # indexes of the first 15 digits of class c\n",
    "    idx <- which(y_train == c)[1:15]\n",
    "    \n",
    "    # prepare plotting area\n",
    "    par(mfcol=c(3, 5))\n",
    "    par(mar=c(0, 0, 0, 0), xaxs = 'i', yaxs = 'i')\n",
    "    \n",
    "    # plot digits corresponding to indexes\n",
    "    for (i in idx) {\n",
    "      img <- x_train[i,,]\n",
    "      img <- t(apply(img, 2, rev))\n",
    "      image(1:28, 1:28, img, col = gray((0:255) / 255), xaxt = 'n', yaxt = 'n')\n",
    "      }\n",
    "    } else {\n",
    "      return(\"Labels are between 0 and 9.\")\n",
    "    }\n",
    "}"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5a3b4205",
   "metadata": {
    "papermill": {
     "duration": 0.006645,
     "end_time": "2023-05-26T14:48:23.227778",
     "exception": false,
     "start_time": "2023-05-26T14:48:23.221133",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Preprocessing\n",
    "As color values are in the [0, 255] interval, we can scale them to be in the [0, 1] interval. Moreover, we can reshape the input by flattening images from a 2D 28 x 28 to 1D 784 (28*28) without information loss:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "99d41b31",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-26T14:48:23.243949Z",
     "iopub.status.busy": "2023-05-26T14:48:23.242584Z",
     "iopub.status.idle": "2023-05-26T14:48:25.036087Z",
     "shell.execute_reply": "2023-05-26T14:48:25.034330Z"
    },
    "papermill": {
     "duration": 1.804825,
     "end_time": "2023-05-26T14:48:25.039222",
     "exception": false,
     "start_time": "2023-05-26T14:48:23.234397",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "x_train_reshaped = array_reshape(x_train, c(60000, 28*28)) / 255\n",
    "x_test_reshaped = array_reshape(x_test, c(10000, 28*28)) / 255"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "abdb41aa",
   "metadata": {
    "papermill": {
     "duration": 0.006909,
     "end_time": "2023-05-26T14:48:25.053149",
     "exception": false,
     "start_time": "2023-05-26T14:48:25.046240",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Keras (as other common libraries) expects to reshape an array by filling new axes in row-major ordering (from the C language). This is the behaviour of array_reshape(). R practitioners may be more familiar with dim<-() to deal with matrices shapes. Nevertheless, dim<-() fills new axes in column-major ordering (from the Fortran language). The labels must be converted from a vector with integers (each integer representing a category) into a matrix with binary values and columns equal to the number of categories:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "98249692",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-26T14:48:25.070185Z",
     "iopub.status.busy": "2023-05-26T14:48:25.068824Z",
     "iopub.status.idle": "2023-05-26T14:48:25.103900Z",
     "shell.execute_reply": "2023-05-26T14:48:25.102175Z"
    },
    "papermill": {
     "duration": 0.046736,
     "end_time": "2023-05-26T14:48:25.106811",
     "exception": false,
     "start_time": "2023-05-26T14:48:25.060075",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "y_train <- to_categorical(y_train)\n",
    "y_test <- to_categorical(y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2659b3ea",
   "metadata": {
    "papermill": {
     "duration": 0.007409,
     "end_time": "2023-05-26T14:48:25.121808",
     "exception": false,
     "start_time": "2023-05-26T14:48:25.114399",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Building the neural network\n",
    "5.1 Define the layers\n",
    "The cornerstone of a neural network is the layer. We can imagine the layer as a module that extracts a representation of the input data that is useful to the final goal.\n",
    "\n",
    "We can build a neural network by stacking layers sequentially. Keras allows to do it by leveraging keras_model_sequential. In this example, we create a network composed of three layers:\n",
    "\n",
    "A fully connected (or dense) layer that produces an output space of 512 units.\n",
    "A dropout layer to “drop out” randomly 20% of neurons during the training. In brief, this technique aims at improving the generalization capabilities of the model.\n",
    "A final dense layer with an output of 10 units and a softmax activation function. This layer returns an array of probability scores for each category, each being the probability of the current image to represent a 0, 1, 2, … up to 9:\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "00d01d5e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-26T14:48:25.138491Z",
     "iopub.status.busy": "2023-05-26T14:48:25.137222Z",
     "iopub.status.idle": "2023-05-26T14:48:28.752783Z",
     "shell.execute_reply": "2023-05-26T14:48:28.751051Z"
    },
    "papermill": {
     "duration": 3.626466,
     "end_time": "2023-05-26T14:48:28.755585",
     "exception": false,
     "start_time": "2023-05-26T14:48:25.129119",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "model <- keras_model_sequential(input_shape = c(28 * 28)) %>%\n",
    "  layer_dense(units = 512, activation = \"relu\") %>%\n",
    "  layer_dropout(0.2) %>%\n",
    "  layer_dense(units = 10, activation = \"softmax\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ddb644c2",
   "metadata": {
    "papermill": {
     "duration": 0.006866,
     "end_time": "2023-05-26T14:48:28.769305",
     "exception": false,
     "start_time": "2023-05-26T14:48:28.762439",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "We can inspect the model’s structure as follows:\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "4c0e565b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-26T14:48:28.785173Z",
     "iopub.status.busy": "2023-05-26T14:48:28.783945Z",
     "iopub.status.idle": "2023-05-26T14:48:28.800528Z",
     "shell.execute_reply": "2023-05-26T14:48:28.798686Z"
    },
    "papermill": {
     "duration": 0.02693,
     "end_time": "2023-05-26T14:48:28.802839",
     "exception": false,
     "start_time": "2023-05-26T14:48:28.775909",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "________________________________________________________________________________\n",
      " Layer (type)                       Output Shape                    Param #     \n",
      "================================================================================\n",
      " dense_1 (Dense)                    (None, 512)                     401920      \n",
      "                                                                                \n",
      " dropout (Dropout)                  (None, 512)                     0           \n",
      "                                                                                \n",
      " dense (Dense)                      (None, 10)                      5130        \n",
      "                                                                                \n",
      "================================================================================\n",
      "Total params: 407,050\n",
      "Trainable params: 407,050\n",
      "Non-trainable params: 0\n",
      "________________________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model %>%\n",
    "  summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ea21ad53",
   "metadata": {
    "papermill": {
     "duration": 0.006798,
     "end_time": "2023-05-26T14:48:28.816328",
     "exception": false,
     "start_time": "2023-05-26T14:48:28.809530",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Compile\n",
    "In the compilation step, we need to define:\n",
    "\n",
    "Loss function:\n",
    "- The loss function must provide a reasonable estimate of the model error.\n",
    "- The network tries to minimize this function during training.\n",
    "Optimizer:\n",
    "- It specifies how the weights of the model get updated during training.\n",
    "- It makes use of the gradient of the loss function.\n",
    "Metrics:\n",
    "- An array of metrics to monitor during the training procedure.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "35fd9d83",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-26T14:48:28.832322Z",
     "iopub.status.busy": "2023-05-26T14:48:28.831075Z",
     "iopub.status.idle": "2023-05-26T14:48:28.854212Z",
     "shell.execute_reply": "2023-05-26T14:48:28.852834Z"
    },
    "papermill": {
     "duration": 0.033691,
     "end_time": "2023-05-26T14:48:28.856694",
     "exception": false,
     "start_time": "2023-05-26T14:48:28.823003",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "model %>% compile(\n",
    "  optimizer = \"rmsprop\",\n",
    "  loss = \"categorical_crossentropy\",\n",
    "  metrics = c(\"accuracy\")\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cdb448dc",
   "metadata": {
    "papermill": {
     "duration": 0.00667,
     "end_time": "2023-05-26T14:48:28.870038",
     "exception": false,
     "start_time": "2023-05-26T14:48:28.863368",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Fit\n",
    "Fitting the model means finding a set of parameters that minimizes the loss function during training.\n",
    "\n",
    "Input data is not processed as a whole. The model iterates over the training data in batches, each of size batch_size. An iteration over all the training data is called epoch. We must declare the number of epochs when fitting the model. After each epoch, the network updates its weights to minimize the loss.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "aec623fb",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-26T14:48:28.886238Z",
     "iopub.status.busy": "2023-05-26T14:48:28.884991Z",
     "iopub.status.idle": "2023-05-26T14:48:41.012471Z",
     "shell.execute_reply": "2023-05-26T14:48:41.010218Z"
    },
    "papermill": {
     "duration": 12.13951,
     "end_time": "2023-05-26T14:48:41.016386",
     "exception": false,
     "start_time": "2023-05-26T14:48:28.876876",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\np1yX4wMAAAAASUVORK5CYII="
     },
     "metadata": {
      "image/png": {
       "height": 420,
       "width": 420
      }
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# fit the model\n",
    "history  <- model %>% fit(\n",
    "  x_train_reshaped,\n",
    "  y_train,\n",
    "  epochs = 3,\n",
    "  batch_size = 128,\n",
    "  validation_split = 0.2\n",
    ")\n",
    "\n",
    "# plot training history\n",
    "history %>% \n",
    "  plot() + \n",
    "  geom_point(size = 3) + \n",
    "  geom_line(linetype = \"dashed\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9fa5b264",
   "metadata": {
    "papermill": {
     "duration": 0.007618,
     "end_time": "2023-05-26T14:48:41.031966",
     "exception": false,
     "start_time": "2023-05-26T14:48:41.024348",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Test set performances\n",
    "After training an estimator, it is a good practice to assess its performances on out-of-sample data. We can measure the accuracy (fraction of handwritten digits correctly classified) on the test set:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "8716b4ac",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-26T14:48:41.049853Z",
     "iopub.status.busy": "2023-05-26T14:48:41.048571Z",
     "iopub.status.idle": "2023-05-26T14:48:41.940057Z",
     "shell.execute_reply": "2023-05-26T14:48:41.937937Z"
    },
    "papermill": {
     "duration": 0.903315,
     "end_time": "2023-05-26T14:48:41.942835",
     "exception": false,
     "start_time": "2023-05-26T14:48:41.039520",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<strong>accuracy:</strong> 0.975"
      ],
      "text/latex": [
       "\\textbf{accuracy:} 0.975"
      ],
      "text/markdown": [
       "**accuracy:** 0.975"
      ],
      "text/plain": [
       "accuracy \n",
       "   0.975 "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "test_metrics <- model %>% \n",
    "  evaluate(x_test_reshaped, y_test)\n",
    "\n",
    "test_metrics[\"accuracy\"] %>% round(., 3)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e7215e7b",
   "metadata": {
    "papermill": {
     "duration": 0.007844,
     "end_time": "2023-05-26T14:48:41.958866",
     "exception": false,
     "start_time": "2023-05-26T14:48:41.951022",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Let us have a look at some misclassified digits:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "c777f7ec",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-26T14:48:41.977528Z",
     "iopub.status.busy": "2023-05-26T14:48:41.976131Z",
     "iopub.status.idle": "2023-05-26T14:48:42.991353Z",
     "shell.execute_reply": "2023-05-26T14:48:42.988850Z"
    },
    "papermill": {
     "duration": 1.027555,
     "end_time": "2023-05-26T14:48:42.994207",
     "exception": false,
     "start_time": "2023-05-26T14:48:41.966652",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\nkZw7d86ZJroIV+wAAABMgo4dAACASdCxAwAAMAkPL1CsE8tL3nJ914EDB4rktttuE8mRI0dE\n0rRp06xP/fvvv+vhZ599lvmhPkhIX/8W+U63bt1E8swzz4hkxowZDo+TnbWpfdlzzz0nEn1E\nXXb89ttvIhk9erRI1q9f7/A4+rhJve7+/v45bJ1St7q5uD4KcNKkSU4c2awqVaokkttvv10k\n7du3F0nfvn1Fkp1RktnZRl+8Wr/R+7Jly0Sij9rcsWOHw3MhL4SGhork559/FsmxY8dE0qVL\nF5Hoi6j36tVLJBUqVBAJY+wAAADgAnTsAAAATIKOHQAAgEnQsQMAADAJr5s8IRZ4XL58ub6N\nHuqD1vUVgPUJFoI+30IXEhLicBudvi4iCxR7ud27d3u6Cfmevjhz586dXXLkjz76SCTZmSqh\nGzJkiEi6du0qktKlSztxZJ0+78qX6dNWFi5cKBJ90Vdv07p1a5EsXbpUJHXr1hVJbGxsHrbJ\nV+k/mvXvBH2qhP6NFBcXJxJ9gWK9q/Dnn39ms53uwRU7AAAAk6BjBwAAYBJ07AAAAEyCjh0A\nAIBJeN3kCedkZy5CduZG5JGaNWuKZMOGDZ5oiPdq2bKlw202btyYR2d/+umnRTJixAiRWCwW\nh8fx8+M3pf+pWrWqSEqVKuXEcf766y+R3HJOlaAPz3/qqadEMmXKFJHo9xtwjv5O2Lx5s0uO\nbA6FChUSSUxMjBPH0e/rsG/fPod7derUSSQFCxZ04uz63QUmT54skvT0dCeOjJwS94hSStWr\nV08kd9xxh0iuXLkikoYNG4pEv9+J/h5LTU3NZjvdg59DAAAAJkHHDgAAwCTo2AEAAJiEh8fY\nFStWTCS//PJL5of6eo9KqRkzZojk+PHjLm2Xiy1evNjTTfAu+lgrvdCbNm0Sif5uyY4uXbqI\nRB/PV7x4cZH4+/uLxDAMkezcuVMk+vK2vrwStf6K6Ul2VKlSRSTjx48Xif7K6xXUlzHXR9Q5\n10JdmTJlROLL7wTdyZMnRaJ/TiMjIx0e5/r16w4T3TPPPCMS/WdKduzdu1ck06dPd+I4yKnb\nb79dJD179hSJPupO/8bWPfjggw630UdSehuu2AEAAJgEHTsAAACToGMHAABgEnTsAAAATMLD\nkyeio6NF8vnnn2d++NZbb+l76UuPvv322yLx8ukUPk4f2F6gQAGRdOzYUST333+/S86uLzWs\nD5m/du2aSF577TWRLFu2TCT6gqXIPf29MXDgQI+0xC45OVkkb775pkjOnDnjruaYhP6qXrx4\nMY/OJaboqVtNuYiIiBCJPtWmWrVqIilZsqRI+E7IvbCwMJF8+umnIrl06ZJI9ClWOn0R9f79\n+4tk3bp1IlmwYIHDI3sWV+wAAABMgo4dAACASdCxAwAAMAkPj7HTb5D83//+N/PDxx9/XN9r\nwIABImnRooVI9IUiz549m/nhkiVLst/ODI0aNXK4zYEDB0SSkJDgxLlMTK+7PqZNH+PiKqdP\nnxaJfpv5adOmiWT9+vV51B6z+u2330Ty999/i0RffNid9HeCPo5Kp4/veeedd1zWJuQ9/Sbu\n+pjs2rVri0QfiVu4cGGR6ONBkXv16tUTSZ06dUSycOFCkVy4cEEkVqvs8Lz77rsiiYqKEsmU\nKVNEov/88jZcsQMAADAJOnYAAAAmQccOAADAJOjYAQAAmIRcqdXblCtXTg9Xr14tEn2ZQV1a\nWlrmh/po/ezQR/Tra+3qczvEjBDoWrZsKZIGDRo43Gvw4MEi2bBhg0j27NkjkqlTp+ascXAR\nfZHniRMnOtwrOwtK62bPni0S3gn4Ny+//LJIJk+eLBL9Xbdjxw6RfPzxxyL58ssvc906X7d0\n6VKRdO7cWSQNGzYUif4jXl9quGzZsiJ54403RJKd7yhvwxU7AAAAk6BjBwAAYBJ07AAAAEyC\njh0AAIBJePvkiVvSBzyOGzdOJI899ph7GqMvYt68eXORXLlyxT2NAQDkVNGiRUWiD7SvUaOG\nw+PoX/XVqlUTSXR0dA5b5+v0aU+1atUSiX73lx49eoikUqVKIvnss89Eok/IS0lJyWY7vQdX\n7AAAAEyCjh0AAIBJ0LEDAAAwiXw5xk6nL2FqtVpF0rdv38wPK1asKDZ4/PHH9SMfPXo088P9\n+/eLDUaPHi2SS5cuZdlYAIBX69Spk0i6d+8uktDQUJHoY/W6du0qkvj4+Fy3zrc888wzItEX\ngtbpyxovXrxYJF9//bVIUlNTc9g6b8QVOwAAAJOgYwcAAGDDSuRHAAAgAElEQVQSdOwAAABM\ngo4dAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\nAAAAvJDFMAxPtwEAAAAu4OfpBgAAAMA1rJ5uAAAApmRLTzeUUhZ/fy6i+BIP190dJx1dPtKS\n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\nAAAASUVORK5CYII="
     },
     "metadata": {
      "image/png": {
       "height": 420,
       "width": 420
      }
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# get test set predictions\n",
    "y_hat <- model %>%\n",
    "  predict(x_test_reshaped) %>% \n",
    "  k_argmax() %>%\n",
    "  as.array()\n",
    "\n",
    "# get test set real labels\n",
    "y_obs <- y_test %>% \n",
    "   k_argmax() %>%\n",
    "   as.array()\n",
    "\n",
    "# get misclassified samples indexes\n",
    "misclass_idx = which(y_hat != y_obs)\n",
    "\n",
    "# plot 25 misclassified digits\n",
    "par(mfcol=c(5, 5))\n",
    "par(mar=c(0, 0, 2.5, 0), xaxs = 'i', yaxs = 'i')\n",
    "    \n",
    "for (i in misclass_idx[0:25]) {\n",
    "      img <- x_test[i,,]\n",
    "      img <- t(apply(img, 2, rev))\n",
    "      image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',\n",
    "        main = paste(\"Predicted: \", y_hat[i] , \"\\nTrue: \", y_obs[i]))\n",
    "}"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cebc982f",
   "metadata": {
    "papermill": {
     "duration": 0.008595,
     "end_time": "2023-05-26T14:48:43.011612",
     "exception": false,
     "start_time": "2023-05-26T14:48:43.003017",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Conclusions\n",
    "In this post, we created a simple neural network in Keras, sharing at the same time an introduction to Deep Learning concepts. In particular, we used the R language, that is generally not as commonly seen in Deep Learning tutorials or guides as Python.\n",
    "\n",
    "Notably, some the two most popular Deep Learning frameworks, Torch and TensorFlow, support R as well.\n",
    "\n",
    "It is possible to find more information and examples here:\n",
    "\n",
    "TensorFlow for R⁶\n",
    "Torch for R⁷\n",
    "Francois Chollet, J.J. Allaire, “Deep Learning with R”, Manning, 2018⁸."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4145c955",
   "metadata": {
    "papermill": {
     "duration": 0.008491,
     "end_time": "2023-05-26T14:48:43.028690",
     "exception": false,
     "start_time": "2023-05-26T14:48:43.020199",
     "status": "completed"
    },
    "tags": []
   },
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "R",
   "language": "R",
   "name": "ir"
  },
  "language_info": {
   "codemirror_mode": "r",
   "file_extension": ".r",
   "mimetype": "text/x-r-source",
   "name": "R",
   "pygments_lexer": "r",
   "version": "4.0.5"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 43.347663,
   "end_time": "2023-05-26T14:48:44.158433",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2023-05-26T14:48:00.810770",
   "version": "2.4.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
