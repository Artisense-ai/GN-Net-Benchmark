# GN-Net Benchmark 

For more information see
[https://vision.in.tum.de/gn-net](https://vision.in.tum.de/gn-net)

This code allows to generate pixel-correspondences for the GN-Net benchmark, presented in
```
"GN-Net: The Gauss-Newton Loss for Multi-Weather Relocalization".
L. von Stumberg, P. Wenzel, Q. Khan and D. Cremers. RA-L 2020
```
 
 ## Install Dependencies
```
pip install opencv-python
pip install scipy==1.1.0
pip install scikit-image
```

## Running

See example.py how to generate matches. At the moment matches can only be generated for the Carla sequences.
For generating the matches either the groundtruth-depths or depths exported from Stereo DSO can be used. 
To specify this you can adjust the parameter ```use_dso_depths``` of the ```generate_matches_carla```

For a start you can simply run
```
python example.py
```