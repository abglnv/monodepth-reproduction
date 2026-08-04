wget -i kitti_archives_to_download.txt -P kitti_data/

cd kitti_data && unzip "*.zip" && cd ..