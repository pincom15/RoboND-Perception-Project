## Project: Perception Pick & Place

---


# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify). 
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  [See the example `output.yaml` for details on what the output should look like.](https://github.com/udacity/RoboND-Perception-Project/blob/master/pr2_robot/config/output.yaml)  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

# Extra Challenges: Complete the Pick & Place
7. To create a collision map, publish a point cloud to the `/pr2/3d_map/points` topic and make sure you change the `point_cloud_topic` to `/pr2/3d_map/points` in `sensors.yaml` in the `/pr2_robot/config/` directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory.  Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
8. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
9. Rotate the robot back to its original state.
10. Create a ROS Client for the “pick_place_routine” rosservice.  In the required steps above, you already created the messages you need to use this service. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
11. If everything was done correctly, when you pass the appropriate messages to the `pick_place_routine` service, the selected arm will perform pick and place operation and display trajectory in the RViz window
12. Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
13. Looking for a bigger challenge?  Load up the `challenge.world` scenario and see if you can get your perception pipeline working there!

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

[//]: # (Image References)

[world1]: ./misc_images/world1.jpg
[world2]: ./misc_images/world2.jpg
[world3]: ./misc_images/world3.jpg
[confusion]: ./misc_images/confusion_matrix.jpg

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

First, we need to convert ROS PointCloud2 message to PCL PointXYGRGB.

```python
    # Convert ROS msg to PCL data
    cloud_filtered = ros_to_pcl(pcl_msg)
```

##### Statistical Outlier Filtering

Now, we can apply a statistical outlier filter to remove RGB-D senser noise from point cloud. PCL's StatisticalOutlierRemoval filter is one of the filtering techniques to remove outliers. It measures the distance of neighbors of each point in the point cloud, and then calculates a mean distance. Points are removed from the point cloud if they are outside of an interval decided by the global distances mean standard deviation. In this project, we will set the number of neighboring points to 3.

```python
    # Statistical Outlier Filtering
    # Much like the previous filters, we start by creating a filter object: 
    outlier_filter = cloud_filtered.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(3)

    # Set threshold scale factor
    x = 0.05

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)

    # Finally call the filter function for magic
    cloud_filtered = outlier_filter.filter()
```

##### Voxel Grid Downsampling

The advantage of downsample, which is reducing the number of points in a point cloud, is to decrease computation. VoxelGrid Downsampling Filter can be used. The units of leaf size are in meters Thus, setting it to 1 means our voxel is 1 cubic meter in volume. We set our leaf size to 0.01, which means our voxels are one centimeter apart.

```python
    # Voxel Grid Downsampling
    vox = cloud_filtered.make_voxel_grid_filter()

    # Choose a voxel (also known as leaf) size (0.003 ~ 0.005)
    LEAF_SIZE = 0.01

    # Set the voxel (or leaf) size  
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()
```

##### PassThrough Filter

We can remove useless data from a point cloud in case we know the location of our target in the scene by using a Pass Throught Filter. The region we pass is often referred to as region of interest. We will apply a Pass Through Filter along 'z' and 'y' axises. By doing this, we can select a region of interest and remove unnecessary data.

```python
    # PassThrough Filter
    # Create a PassThrough filter object.
    passthrough = cloud_filtered.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.61
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)

    # Finally use the filter function to obtain the resultant point cloud. 
    cloud_filtered = passthrough.filter()

    passthrough = cloud_filtered.make_passthrough_filter()

    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.4
    axis_max =  0.4
    passthrough.set_filter_limits(axis_min, axis_max)

    # Use the filter function to obtain the resultant point cloud. 
    cloud_filtered = passthrough.filter()
```

##### RANSAC Plane Segmentation

We can remove the table itself from the secne using Random Sample Consensus or "RANSAC". RANSAC algrithm can be used to classify points in a point cloud that belong to a particular model. the model can be a plane, a cylinder, a box, or any other common shape in 3D scene. RANSAC assumes that the data consists of inliers whose distrubution fits a particular model, and outliers that does not fit the model. Using RANSAC, we can remove the inliers from the point cloud that are good fits for the model.

```python
    # RANSAC Plane Segmentation
    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()

    # Set the model you wish to fit 
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Set the model you wish to fit 
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance 
    # for segmenting the table
    max_distance = 0.006
    seg.set_distance_threshold(max_distance)

    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    cloud_objects = cloud_filtered.extract(inliers, negative=True)
    cloud_table = cloud_filtered.extract(inliers, negative=False)
```

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.

##### Euclidean Clustering

With PCL's Euclidean Clustering algorithm, we can segment the points, which are filtered out the table, and outside of the region of interest , into individual objects. Before we perform Euclidean Clustering, we need to construct a k-d tree to reduce computation. Once we have k-d tree, we can extract the cluster.

```python
    # Euclidean Clustering
    # Apply function to convert XYZRGB to XYZ
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(0.03)
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(9000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()
```


##### Cluster Visualization

We can visualize the lists of points for each object in RViz.

```python
    #Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                white_cloud[indice][1],
                white_cloud[indice][2],
                rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
```

##### Convert PCL data to ROS messages and publish them.

```python
    # Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)
```

```python
    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)
```

#### 3. Complete Exercise 3 Steps. Features extracted and SVM trained. Object recognition implemented.

We can classify each object using Support Vector Machine or "SVM". SVM is a particular supervised learning algrithm that analyze data for classification and regression analysis. Given labeled training data, we can characterize the point cloud into discrete classes.

Before we use SVM, we need to generate features. Following codes are used to capture features.

```python
def compute_color_histograms(cloud, using_hsv=True):

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])

    # TODO: Compute histograms
    nbins=64
    bins_range=(0, 256)

    # Compute the histogram of the channels separately
    channel_1_hist = np.histogram(channel_1_vals, bins=nbins, range=bins_range)
    channel_2_hist = np.histogram(channel_2_vals, bins=nbins, range=bins_range)
    channel_3_hist = np.histogram(channel_3_vals, bins=nbins, range=bins_range)

    # TODO: Concatenate and normalize the histograms
    hist_features = np.concatenate((channel_1_hist[0], channel_2_hist[0], channel_1_hist[0])).astype(np.float64)

    # Normalize the result
    normed_features = hist_features / np.sum(hist_features)

    return normed_features
```

```python
def compute_normal_histograms(normal_cloud):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    # TODO: Compute histograms of normal values (just like with color)
    nbins=40
    bins_range=(-1., 1.)

    # Compute the histogram of the channels separately
    norm_x_hist = np.histogram(norm_x_vals, bins=nbins, range=bins_range)
    norm_y_hist = np.histogram(norm_y_vals, bins=nbins, range=bins_range)
    norm_z_hist = np.histogram(norm_z_vals, bins=nbins, range=bins_range)

    # TODO: Concatenate and normalize the histograms
    hist_features = np.concatenate((norm_x_hist[0], norm_y_hist[0], norm_z_hist[0])).astype(np.float64)

    # Normalize the result
    normed_features = hist_features / np.sum(hist_features)

    return normed_features

```

In order to generate features, we need to launch the `training.launch` file to open the Gazebe environment.

```sh
cd ~/catkin_ws
roslaunch sensor_stick training.launch
```

In a new terminal, we can run `capture_features.py` script to capture and save features for each of the objects.

```
$ cd ~/catkin_ws
$ rosrun sensor_stick capture_features.py
```

Once we extract features, we are ready to train our model.

```sh
rosrun sensor_stick train_svm.py
```

![confusion][confusion]

```sh
Features in Training Set: 400
Invalid Features in Training set: 1
Scores: [ 0.9625      0.9125      0.9125      0.9         0.96202532]
Accuracy: 0.93 (+/- 0.05)
accuracy score: 0.929824561404
```

After we obtained a trained classifier, we can do object recognition.

```python
    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)

        # Compute the associated feature vector
        # convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        # complete this step just as is covered in capture_features.py
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)
```

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

##### Test1

![world1][world1]

##### Test2

![world2][world2]

##### Test3

![world3][world3]

##### Output yaml files

[output_1.yaml](./pr2_robot/scripts/output/output_1.yaml)

[output_2.yaml](./pr2_robot/scripts/output/output_2.yaml)

[output_3.yaml](./pr2_robot/scripts/output/output_3.yaml)

##### Future Work

To improve this project further, increasing the numbers of samples in `capture_features.ph`. Also, it would be better to run this project on more powerful computer in a native Linux, instead of VM.

Also, it would be a good try that we adjust configurations such as `nbins` in `features.py`.

