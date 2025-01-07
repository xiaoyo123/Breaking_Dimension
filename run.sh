python yolact_ori/eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=input/photo.png:results/photo.png
python resize.py

python TripoSR/run.py results/photo.png --output-dir results/

#yolact head
python yolact/evalHead.py --trained_model=weights/yolact_base_2933_17600.pth --score_threshold=0.15 --top_k=15 --image=results/photo.png:results/photo.png

python cut.py

/home/xiaoyo/blender-4.1.1-linux-x64/blender --background --python blenderScript/blenderScript.py 

python png2gif.py
