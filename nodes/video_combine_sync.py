"""
Video Combine Sync for ComfyUI

Copy of Video Helper Suite's "Video Combine 🎥🅥🅗🅢" with one addition:
a boolean input that stretches the audio to match the final video duration
using ffmpeg's atempo filter, which changes tempo without shifting pitch.
(VAE/latent input support from the original is dropped.)

Requires the comfyui-videohelperssuite package to be installed.
"""

import datetime
import itertools
import json
import os
import re
import subprocess
import sys

import torch
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.utils import ProgressBar

# videohelpersuite lives inside its custom node folder, not on sys.path
_VHS_DIR = None
try:
    for custom_nodes_dir in folder_paths.get_folder_paths("custom_nodes"):
        candidate = os.path.join(custom_nodes_dir, "comfyui-videohelpersuite")
        if os.path.isdir(candidate):
            _VHS_DIR = candidate
            break
except Exception:
    pass

if _VHS_DIR is None:
    _VHS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "custom_nodes", "comfyui-videohelpersuite")

if os.path.isdir(_VHS_DIR) and _VHS_DIR not in sys.path:
    sys.path.insert(0, _VHS_DIR)

try:
    from videohelpersuite.nodes import get_video_formats, apply_format_widgets, ContainsAll, \
            imageOrLatent, floatOrInt, to_pingpong, tensor_to_bytes, tensor_to_shorts, gifski_process, \
            ffmpeg_process
    from videohelpersuite.utils import ffmpeg_path, merge_filter_args, requeue_workflow, ENCODE_ARGS
    HAS_VHS = True
except Exception:
    HAS_VHS = False
    imageOrLatent = "IMAGE"
    floatOrInt = "FLOAT"
    def get_video_formats():
        return [], {}
    def apply_format_widgets(*args, **kwargs):
        pass
    class ContainsAll:
        pass
    def to_pingpong(*args):
        pass
    def tensor_to_bytes(*args):
        pass
    def tensor_to_shorts(*args):
        pass
    def gifski_process(*args):
        pass
    def ffmpeg_process(*args):
        pass
    def ffmpeg_path():
        return "ffmpeg"
    def merge_filter_args(*args):
        return []
    def requeue_workflow(*args):
        pass
    ENCODE_ARGS = ("utf-8", "ignore")





def _atempo_chain(tempo):
    """Build an atempo filter chain; each filter is limited to [0.5, 2.0]."""
    parts = []
    t = tempo
    while t > 2.0:
        parts.append("atempo=2.0")
        t /= 2.0
    while t < 0.5:
        parts.append("atempo=0.5")
        t *= 2.0
    parts.append(f"atempo={t:.6f}")
    return ",".join(parts)


class VideoCombineSync:
    @classmethod
    def INPUT_TYPES(s):
        ffmpeg_formats, format_widgets = get_video_formats()
        format_widgets["image/webp"] = [['lossless', "BOOLEAN", {'default': True}]]
        return {
            "required": {
                "images": (imageOrLatent,),
                "frame_rate": (
                    floatOrInt,
                    {"default": 8, "min": 1, "step": 1},
                ),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
                "format": (["image/gif", "image/webp"] + ffmpeg_formats, {'formats': format_widgets}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "match_audio_to_video": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Stretch the audio to the video duration with pitch compensation (atempo), instead of padding/cutting it.",
                }),
                "meta_batch": ("VHS_BatchManager",),
            },
            "hidden": ContainsAll({
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            }),
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "combine_video"

    def combine_video(
        self,
        frame_rate: int,
        loop_count: int,
        images=None,
        filename_prefix="AnimateDiff",
        format="image/gif",
        pingpong=False,
        save_output=True,
        prompt=None,
        extra_pnginfo=None,
        audio=None,
        match_audio_to_video=False,
        unique_id=None,
        manual_format_widgets=None,
        meta_batch=None,
        **kwargs
    ):
        if images is None:
            return ((save_output, []),)
        if isinstance(images, torch.Tensor) and images.size(0) == 0:
            return ((save_output, []),)
        num_frames = len(images)
        pbar = ProgressBar(num_frames)
        first_image = images[0]
        images = iter(images)
        # get output information
        output_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)
        output_files = []

        metadata = PngInfo()
        video_metadata = {}
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
            video_metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                video_metadata[x] = extra_pnginfo[x]
            extra_options = extra_pnginfo.get('workflow', {}).get('extra', {})
        else:
            extra_options = {}
        metadata.add_text("CreationTime", datetime.datetime.now().isoformat(" ")[:19])

        if meta_batch is not None and unique_id in meta_batch.outputs:
            (counter, output_process) = meta_batch.outputs[unique_id]
        else:
            # comfy counter workaround
            max_counter = 0

            # Loop through the existing files
            matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
            for existing_file in os.listdir(full_output_folder):
                # Check if the file matches the expected format
                match = matcher.fullmatch(existing_file)
                if match:
                    # Extract the numeric portion of the filename
                    file_counter = int(match.group(1))
                    # Update the maximum counter value if necessary
                    if file_counter > max_counter:
                        max_counter = file_counter

            # Increment the counter by 1 to get the next available value
            counter = max_counter + 1
            output_process = None

        # save first frame as png to keep metadata
        first_image_file = f"{filename}_{counter:05}.png"
        file_path = os.path.join(full_output_folder, first_image_file)
        if extra_options.get('VHS_MetadataImage', True) != False:
            Image.fromarray(tensor_to_bytes(first_image)).save(
                file_path,
                pnginfo=metadata,
                compress_level=4,
            )
        output_files.append(file_path)

        format_type, format_ext = format.split("/")
        if format_type == "image":
            if meta_batch is not None:
                raise Exception("Pillow('image/') formats are not compatible with batched output")
            image_kwargs = {}
            if format_ext == "gif":
                image_kwargs['disposal'] = 2
            if format_ext == "webp":
                #Save timestamp information
                exif = Image.Exif()
                exif[ExifTags.IFD.Exif] = {36867: datetime.datetime.now().isoformat(" ")[:19]}
                image_kwargs['exif'] = exif
                image_kwargs['lossless'] = kwargs.get("lossless", True)
            file = f"{filename}_{counter:05}.{format_ext}"
            file_path = os.path.join(full_output_folder, file)
            if pingpong:
                images = to_pingpong(images)
            def frames_gen(images):
                for i in images:
                    pbar.update(1)
                    yield Image.fromarray(tensor_to_bytes(i))
            frames = frames_gen(images)
            # Use pillow directly to save an animated image
            next(frames).save(
                file_path,
                format=format_ext.upper(),
                save_all=True,
                append_images=frames,
                duration=round(1000 / frame_rate),
                loop=loop_count,
                compress_level=4,
                **image_kwargs
            )
            output_files.append(file_path)
        else:
            # Use ffmpeg to save a video
            if ffmpeg_path is None:
                raise ProcessLookupError(f"ffmpeg is required for video outputs and could not be found.\nIn order to use video outputs, you must either:\n- Install imageio-ffmpeg with pip,\n- Place a ffmpeg executable in {os.path.abspath('')}, or\n- Install ffmpeg and add it to the system path.")

            if manual_format_widgets is not None:
                print("Format args can now be passed directly. The manual_format_widgets argument is now deprecated")
                kwargs.update(manual_format_widgets)

            has_alpha = first_image.shape[-1] == 4
            kwargs["has_alpha"] = has_alpha
            video_format = apply_format_widgets(format_ext, kwargs)
            dim_alignment = video_format.get("dim_alignment", 2)
            if (first_image.shape[1] % dim_alignment) or (first_image.shape[0] % dim_alignment):
                #output frames must be padded
                to_pad = (-first_image.shape[1] % dim_alignment,
                          -first_image.shape[0] % dim_alignment)
                padding = (to_pad[0]//2, to_pad[0] - to_pad[0]//2,
                           to_pad[1]//2, to_pad[1] - to_pad[1]//2)
                padfunc = torch.nn.ReplicationPad2d(padding)
                def pad(image):
                    image = image.permute((2,0,1))#HWC to CHW
                    padded = padfunc(image.to(dtype=torch.float32))
                    return padded.permute((1,2,0))
                images = map(pad, images)
                dimensions = (-first_image.shape[1] % dim_alignment + first_image.shape[1],
                              -first_image.shape[0] % dim_alignment + first_image.shape[0])
                print("Output images were not of valid resolution and have had padding applied")
            else:
                dimensions = (first_image.shape[1], first_image.shape[0])
            if pingpong:
                if meta_batch is not None:
                    raise Exception("pingpong is incompatible with batched output")
                images = to_pingpong(images)
                if num_frames > 2:
                    num_frames += num_frames -2
                    pbar.total = num_frames
            if loop_count > 0:
                loop_args = ["-vf", "loop=loop=" + str(loop_count)+":size=" + str(num_frames)]
            else:
                loop_args = []
            if video_format.get('input_color_depth', '8bit') == '16bit':
                images = map(tensor_to_shorts, images)
                if has_alpha:
                    i_pix_fmt = 'rgba64'
                else:
                    i_pix_fmt = 'rgb48'
            else:
                images = map(tensor_to_bytes, images)
                if has_alpha:
                    i_pix_fmt = 'rgba'
                else:
                    i_pix_fmt = 'rgb24'
            file = f"{filename}_{counter:05}.{video_format['extension']}"
            file_path = os.path.join(full_output_folder, file)
            bitrate_arg = []
            bitrate = video_format.get('bitrate')
            if bitrate is not None:
                bitrate_arg = ["-b:v", str(bitrate) + "M" if video_format.get('megabit') == 'True' else str(bitrate) + "K"]
            args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", i_pix_fmt,
                    "-color_range", "pc", "-colorspace", "rgb", "-color_primaries", "bt709",
                    "-color_trc", video_format.get("fake_trc", "iec61966-2-1"),
                    "-s", f"{dimensions[0]}x{dimensions[1]}", "-r", str(frame_rate), "-i", "-"] \
                    + loop_args

            images = map(lambda x: x.tobytes(), images)
            env=os.environ.copy()
            if  "environment" in video_format:
                env.update(video_format["environment"])

            if "pre_pass" in video_format:
                if meta_batch is not None:
                    #Performing a prepass requires keeping access to all frames.
                    raise Exception("Formats which require a pre_pass are incompatible with Batch Manager.")
                images = [b''.join(images)]
                os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
                in_args_len = args.index("-i") + 2 # The index after ["-i", "-"]
                pre_pass_args = args[:in_args_len] + video_format['pre_pass']
                merge_filter_args(pre_pass_args)
                try:
                    subprocess.run(pre_pass_args, input=images[0], env=env,
                                   capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occurred in the ffmpeg prepass:\n" \
                            + e.stderr.decode(*ENCODE_ARGS))
            if "inputs_main_pass" in video_format:
                in_args_len = args.index("-i") + 2 # The index after ["-i", "-"]
                args = args[:in_args_len] + video_format['inputs_main_pass'] + args[in_args_len:]

            if output_process is None:
                if 'gifski_pass' in video_format:
                    format = 'image/gif'
                    output_process = gifski_process(args, dimensions, frame_rate, video_format, file_path, env)
                    audio = None
                else:
                    args += video_format['main_pass'] + bitrate_arg
                    merge_filter_args(args)
                    output_process = ffmpeg_process(args, video_format, video_metadata, file_path, env)
                #Proceed to first yield
                output_process.send(None)
                if meta_batch is not None:
                    meta_batch.outputs[unique_id] = (counter, output_process)

            for image in images:
                pbar.update(1)
                output_process.send(image)
            if meta_batch is not None:
                requeue_workflow((meta_batch.unique_id, not meta_batch.has_closed_inputs))
            if meta_batch is None or meta_batch.has_closed_inputs:
                #Close pipe and wait for termination.
                try:
                    total_frames_output = output_process.send(None)
                    output_process.send(None)
                except StopIteration:
                    pass
                if meta_batch is not None:
                    meta_batch.outputs.pop(unique_id)
                    if len(meta_batch.outputs) == 0:
                        meta_batch.reset()
            else:
                #batch is unfinished
                return {"ui": {"unfinished_batch": [True]}, "result": ((save_output, []),)}

            output_files.append(file_path)


            a_waveform = None
            if audio is not None:
                try:
                    #safely check if audio produced by VHS_LoadVideo actually exists
                    a_waveform = audio['waveform']
                except:
                    pass
            if a_waveform is not None:
                # Create audio file if input was provided
                output_file_with_audio = f"{filename}_{counter:05}-audio.{video_format['extension']}"
                output_file_with_audio_path = os.path.join(full_output_folder, output_file_with_audio)
                if "audio_pass" not in video_format:
                    print("Selected video format does not have explicit audio support")
                    video_format["audio_pass"] = ["-c:a", "libopus"]

                channels = audio['waveform'].size(1)
                filters = []
                if match_audio_to_video:
                    audio_dur = audio['waveform'].size(-1) / audio['sample_rate']
                    video_dur = total_frames_output / frame_rate
                    if audio_dur > 0 and video_dur > 0:
                        tempo = audio_dur / video_dur
                        if abs(tempo - 1.0) > 0.001:
                            filters.append(_atempo_chain(tempo))
                    # small epsilon pad so -shortest doesn't trim rounding leftovers
                    filters.append(f"apad=whole_dur={video_dur + 0.05}")
                    apad = ["-af", ",".join(filters)] if filters else []
                else:
                    min_audio_dur = total_frames_output / frame_rate + 1
                    apad = ["-af", f"apad=whole_dur={min_audio_dur}"]
                mux_args = [ffmpeg_path, "-v", "error", "-n", "-i", file_path,
                            "-ar", str(audio['sample_rate']), "-ac", str(channels),
                            "-f", "f32le", "-i", "-", "-c:v", "copy"] \
                            + video_format["audio_pass"] \
                            + apad + ["-shortest", output_file_with_audio_path]

                audio_data = audio['waveform'].squeeze(0).transpose(0,1) \
                        .numpy().tobytes()
                merge_filter_args(mux_args, '-af')
                try:
                    res = subprocess.run(mux_args, input=audio_data,
                                         env=env, capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occured in the ffmpeg subprocess:\n" \
                            + e.stderr.decode(*ENCODE_ARGS))
                if res.stderr:
                    print(res.stderr.decode(*ENCODE_ARGS), end="", file=sys.stderr)
                output_files.append(output_file_with_audio_path)
                #Return this file with audio to the webui.
                #It will be muted unless opened or saved with right click
                file = output_file_with_audio
        if extra_options.get('VHS_KeepIntermediate', True) == False:
            for intermediate in output_files[1:-1]:
                if os.path.exists(intermediate):
                    os.remove(intermediate)
        preview = {
                "filename": file,
                "subfolder": subfolder,
                "type": "output" if save_output else "temp",
                "format": format,
                "frame_rate": frame_rate,
                "workflow": first_image_file,
                "fullpath": output_files[-1],
            }
        if num_frames == 1 and 'png' in format and '%03d' in file:
            preview['format'] = 'image/png'
            preview['filename'] = file.replace('%03d', '001')
        return {"ui": {"gifs": [preview]}, "result": ((save_output, output_files),)}


NODE_CLASS_MAPPINGS = {
    "VideoCombineSync": VideoCombineSync,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCombineSync": "Video Combine Sync 🎥🅥🅗🅢",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
