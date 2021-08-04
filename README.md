# reference_based_sketch_image_colorization
PyTorch implementation of the paper "[Reference-Based Sketch Image Colorization using Augmented-Self Reference and Dense Semantic Correspondence](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lee_Reference-Based_Sketch_Image_Colorization_Using_Augmented-Self_Reference_and_Dense_Semantic_CVPR_2020_paper.pdf)" (CVPR 2020)

## Dependencies

- Pytorch
- torchvision
- numpy
- PIL
- OpenCV
- tqdm


## Usage

1. Clone the repository

- ```git clone https://github.com/Snailpong/style_transfer_implementation.git```


2. Dataset download

- Tag2Pix (filtered Danbooru2020): [Link](https://github.com/blandocs/Tag2Pix)
- You need to change the script 'danbooru2018' to 'danbooru2020' (can be changed)
- In my experiment, I used about 6000 images filtered by ```python preprocessor/tagset_extractor.py```
    - I stopped the process when ```0080``` folder was finished downloading.


3. Sketch image generation

- XDoG: [Link](https://github.com/garygrossi/XDoG-Python)
- For automatic genration, I edited main function as follows:
```
if __name__ == '__main__':
  for file_name in os.listdir('../data/danbooru/color'):
      print(file_name, end='\r')
      image = cv2.imread(f'../data/danbooru/color/{file_name}', cv2.IMREAD_GRAYSCALE)
      result = xdog(image)
      cv2.imwrite(f'../data/danbooru/sketch/{file_name}', result)
```

- folder structure example
```
.
└── data
    ├── danbooru
    |   ├── color
    |   |   ├── 7.jpg
    |   |   └── ...
    |   └── sketch
    |       ├── 7.jpg
    |       └── ...
    └── val
        ├── color
        |   ├── 1.jpg
        |   └── ...
        └── sketch
            ├── 1.jpg
            └── ...
```


4. TPS transformation module

- TPS: [Link](https://github.com/cheind/py-thin-plate-spline)
- Place ```thinplate``` folder to main folder


5. Train

- ``` python train.py```
- arguments
    - load_model: ```True```/```False```
    - cuda_visible: ```CUDA_VISIBLE_DEVICES``` (e.g. 1)


6. Test
- ``` python test.py```

- arguments
    - image_path: folder path to convert the images
    - cuda_visible


## Results

<table style="text-align: center">
<tr><td>Sketch</td><td>Reference</td><td>Result</td></tr>
<tr>
<td><img src="https://user-images.githubusercontent.com/11583179/128166077-1450e90e-96af-4217-9f07-38d37c854622.jpg"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/128167543-70b8cf66-077b-4a73-b33a-610ce8f90e7c.jpg"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/128169661-fed793ba-5ab5-425f-bd60-795da6389f61.png"></td>
</tr>
<tr>
<td><img src="https://user-images.githubusercontent.com/11583179/128166080-21c4f800-e60f-454a-a85f-1f48bf012bb3.jpg"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/128167547-fb9f9ea2-7a3a-4c31-ab32-280df64fc610.jpg"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/128169773-0e1df887-24f3-4ae5-bcf0-83da077fa94e.png"></td>
</tr>
<tr>
<td><img src="https://user-images.githubusercontent.com/11583179/128166083-f593081a-7cf4-4532-885f-45f628017706.jpg"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/128167548-5b7b23fe-c2f8-4084-81e0-b55fded2c13e.jpg"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/128171085-2fefdc96-2d9d-4843-b8b0-fff8857a3ace.png"></td>
</tr>
<tr>
<td><img src="https://user-images.githubusercontent.com/11583179/128166085-adff48ae-76ff-4a04-834d-47aa73d2e7d1.jpg"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/128167550-abc9f03a-a9b1-4096-8f2a-d5623bf5c16e.jpg"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/128171214-5dfdbc5f-8e5d-4baf-8433-ef06bc8e1bd9.png"></td>
</tr>
<tr>
<td><img src="https://user-images.githubusercontent.com/11583179/128166087-d2d0dba8-5ae4-467a-baf0-dc5f2ed2940a.jpg"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/128167551-8d749813-c36a-4ed2-a553-ccb80181157b.jpg"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/128171324-97c58632-9e27-42b7-a151-ff03b46b0f9a.png"></td>
</tr>
<tr>
<td><img src="https://user-images.githubusercontent.com/11583179/128166089-2880b749-78df-4460-b2f3-ea869d3c4d67.jpg"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/128167554-f9f4e7e6-0406-425e-be39-ad8237221c55.jpg"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/128171623-b28cb1c8-2441-4aeb-9b10-2a40da5122c3.png"></td>
</tr>
<tr>
<td><img src="https://user-images.githubusercontent.com/11583179/128166092-3022b06d-1e54-48d0-bd19-3c226ae0f087.jpg"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/128167555-4ec1b321-b4d1-4fd0-8bfc-ed70ee1a642f.jpg"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/128171811-1248101b-7236-4368-9f8d-829c37de64d5.png"></td>
</tr>
<tr>
<td><img src="https://user-images.githubusercontent.com/11583179/128166093-87d6551e-8eed-487a-8e2e-cbb40e30fa49.jpg"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/128167556-3ec6a9f8-ac62-4e1d-bfb9-f9b9690102d1.jpg"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/128171948-2e41db31-bb83-41df-bcfa-588518beb65a.png"></td>
</tr>
<tr>
<td><img src="https://user-images.githubusercontent.com/11583179/128166095-2fef5754-dccf-420e-8d41-50323583aac8.jpg"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/128167558-aa69b017-f7a2-4457-aa8d-50b6256600be.jpg"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/128172030-2d2ec53f-1a7e-46c5-bb97-0e3815debf9f.png"></td>
</tr>
<tr>
<td><img src="https://user-images.githubusercontent.com/11583179/128166075-0604b085-621e-44fb-b5ec-3c2f73df9dc8.jpg"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/128190929-ae8cd35b-7b08-4584-86c2-caa0e765f954.png"></td>
<td><img src="https://user-images.githubusercontent.com/11583179/128172120-f1364947-488b-4c7d-bcfa-54c602d62c30.png"></td>
</tr>

</table>

## Observation & Discussion

- In Eq. (1), I could not scale the number of activation map, instead I scaled activation map into <img src="https://render.githubusercontent.com/render/math?math=f^{l_p} \in R^{h_p \times w_p \times c_l}">.
- In Eq. (5), I implemented the negative region as same region in different batches since the negative region is ambiguous.
- In Eq. (9), since <img src="https://render.githubusercontent.com/render/math?math=l"> is unclear in contrast to Eq. (8), I computed style (gram) loss with ```relu5_1``` activation map.
- In this experiment, there was little difference in quality with or without the similarity-based triplet loss. After convergence from 20 to 0 from 1 epoch, there was little change.
- When the test image was predicted every 1 epoch after the content loss was converged, the color quality difference was remarkable.
- The converged adversarial losses of the generator and discriminator were 0.7 ~ 0.8 and 0.15 ~ 0.2, respectively.


## Code Reference

- <https://github.com/blandocs/Tag2Pix>
- <https://github.com/garygrossi/XDoG-Python>
- <https://github.com/cheind/py-thin-plate-spline>
- <https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49>
- <https://github.com/Jungjaewon/Reference_based_Skectch_Image_Colorization>
