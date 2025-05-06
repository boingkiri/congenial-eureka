# pretrained_yolo_path=/mnt/home/jeongjun/layout_diffusion/yolov11/generative_active_learning/250118_ms_coco_stuff_real_data_4p/weights/best.pt
# unseen_file_list_filename=/mnt/home/jeongjun/layout_diffusion/yolov11/ms_coco_training_total_set.txt

pretrained_yolo_path=/mnt/home/jeongjun/layout_diffusion/yolov11/generative_active_learning/250409_ms_coco_stuff_real_data_4p_againagain/weights/best.pt
unseen_file_list_filename=/mnt/home/jeongjun/layout_diffusion/coreset/yolo_coreset/250409_ms_coco_stuff_real_data_4p_againagain/newly_selected_250409_ms_coco_stuff_real_data_4p_againagain.txt
output_filename=250409_ms_coco_stuff_real_data_4p_againagain

cuda_visible_devices=5

CUDA_VISIBLE_DEVICES=$cuda_visible_devices python label.py \
    --pretrained_yolo_path $pretrained_yolo_path \
    --unseen_file_list_filename $unseen_file_list_filename \
    --output_filename $output_filename