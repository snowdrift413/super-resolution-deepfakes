from core.main import display, generate, preprocess

source_image = str(input("Enter the name and type of your source image >> "))
driving_video = str(input("Enter the name and type of your driving video >> "))
cpu = True
improve = False
best_frame = False
relative = True
adapt_movement_scale = True
config = "vox-256.yaml"
checkpoint = "vox-cpk.pth.tar"
source_image, driving_video, fps, image_type = preprocess(source_image=source_image, driving_video=driving_video)
video = generate(source_image=source_image, driving_video=driving_video, image_type=image_type, fps=fps, best_frame=best_frame, relative=relative, adapt_movement_scale=adapt_movement_scale, improve=improve, config=config, checkpoint=checkpoint, cpu=cpu)
