import os
import win32com.client
import subprocess

def convert_ppt_to_images(ppt_path, output_folder):
    # Kill any running PowerPoint instances
    subprocess.run(["taskkill", "/F", "/IM", "POWERPNT.EXE"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open PowerPoint application
    ppt_app = win32com.client.Dispatch("PowerPoint.Application")
    ppt_app.Visible = 1  # Show PowerPoint (Optional)

    # Open the presentation
    presentation = ppt_app.Presentations.Open(os.path.abspath(ppt_path), WithWindow=False)

    slide_images = []
    
    # Export slides as images
    for i, slide in enumerate(presentation.Slides):
        img_filename = f"slide_{i+1}.png"
        img_path = os.path.abspath(os.path.join(output_folder, img_filename))
        slide.Export(img_path, "PNG")
        slide_images.append(img_filename)
        print(f"✅ Slide {i+1} saved as {img_path}")

    # Close PowerPoint
    presentation.Close()
    ppt_app.Quit()

    print(f"✅ Successfully converted {i+1} slides into images.")
    return slide_images  # Return list of images
