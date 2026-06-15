import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.generic")
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")

import types
from typing import Union, Any
import coqpit.coqpit

def is_union_patched(arg_type: Any) -> bool:
    if type(arg_type).__name__ == "UnionType":
        return True
    try:
        return coqpit.coqpit.safe_issubclass(arg_type, Union)
    except AttributeError:
        return False
coqpit.coqpit.is_union = is_union_patched

from TTS.api import TTS
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse,StreamingResponse
from typing import Optional

import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from pydantic import BaseModel
import uvicorn

import os
import time
from pathlib import Path
import shutil
from loguru import logger
from argparse import ArgumentParser
from pathlib import Path
from uuid import uuid4

import asyncio
from fastapi.concurrency import run_in_threadpool

from xtts_api_server.tts_funcs import TTSWrapper,supported_languages,InvalidSettingsError
from xtts_api_server.RealtimeTTS import TextToAudioStream, CoquiEngine
from xtts_api_server.modeldownloader import check_stream2sentence_version,install_deepspeed_based_on_python_version

from xtts_api_server.modeldownloader import check_stream2sentence_version,install_deepspeed_based_on_python_version

def detect_language(text, default="vi"):
    try:
        from langdetect import detect
        lang = detect(text)
        if lang in supported_languages:
            return lang
        # Fallback mappings for some langdetect codes
        if lang == "zh-cn" or lang == "zh-tw": return "zh-cn"
        return default
    except Exception:
        return default

# Default Folders , you can change them via API
DEVICE = os.getenv('DEVICE',"cuda")
OUTPUT_FOLDER = os.getenv('OUTPUT', 'output')
SPEAKER_FOLDER = os.getenv('SPEAKER', 'speakers')
MODEL_FOLDER = os.getenv('MODEL', 'xtts_models')
BASE_HOST = os.getenv('BASE_URL', '127.0.0.1:8020')
BASE_URL = os.getenv('BASE_URL', '127.0.0.1:8020')
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")
MODEL_VERSION = os.getenv("MODEL_VERSION","XTTS-v2-vietnamse")
LOWVRAM_MODE = os.getenv("LOWVRAM_MODE") == 'true'
DEEPSPEED = os.getenv("DEEPSPEED") == 'true'
USE_CACHE = os.getenv("USE_CACHE") == 'true'

# AUDIO PROCESSING VARS
ENABLE_DENOISING = os.getenv("ENABLE_DENOISING") == 'true'
DENOISING_BACKEND = os.getenv("DENOISING_BACKEND", "demucs")
OUTPUT_SAMPLE_RATE_48K = os.getenv("OUTPUT_SAMPLE_RATE_48K") == 'true'
UP_SAMPLER_BACKEND = os.getenv("UP_SAMPLER_BACKEND", "dsp")

# STREAMING VARS
STREAM_MODE = os.getenv("STREAM_MODE") == 'true'
STREAM_MODE_IMPROVE = os.getenv("STREAM_MODE_IMPROVE") == 'true'
STREAM_PLAY_SYNC = os.getenv("STREAM_PLAY_SYNC") == 'true'

if(DEEPSPEED):
  install_deepspeed_based_on_python_version()

# Create an instance of the TTSWrapper class and server
app = FastAPI()
XTTS = TTSWrapper(
    output_folder=OUTPUT_FOLDER,
    speaker_folder=SPEAKER_FOLDER,
    model_folder=MODEL_FOLDER,
    lowvram=LOWVRAM_MODE,
    model_source=MODEL_SOURCE,
    model_version=MODEL_VERSION,
    device=DEVICE,
    deepspeed=DEEPSPEED,
    enable_cache_results=USE_CACHE,
    enable_denoising=ENABLE_DENOISING,
    denoising_backend=DENOISING_BACKEND,
    output_sample_rate_48k=OUTPUT_SAMPLE_RATE_48K,
    up_sampler_backend=UP_SAMPLER_BACKEND
)

tts_queue = asyncio.Semaphore(1)
queue_count = 0
MAX_QUEUE_SIZE = 10

GC_OUTPUT_DAYS = int(os.getenv("GC_OUTPUT_DAYS", 3))
GC_TEMP_DAYS = int(os.getenv("GC_TEMP_DAYS", 1))

async def garbage_collector():
    while True:
        try:
            logger.info("Running Garbage Collection for audio files...")
            now = time.time()
            
            # Clean up output folder
            if os.path.exists(OUTPUT_FOLDER):
                for filename in os.listdir(OUTPUT_FOLDER):
                    file_path = os.path.join(OUTPUT_FOLDER, filename)
                    if os.path.isfile(file_path):
                        try:
                            if os.stat(file_path).st_mtime < now - GC_OUTPUT_DAYS * 86400:
                                os.remove(file_path)
                                logger.info(f"GC deleted old output file: {filename}")
                        except Exception:
                            pass
                            
            # Clean up speakers folder (only temp_ files)
            if os.path.exists(SPEAKER_FOLDER):
                for filename in os.listdir(SPEAKER_FOLDER):
                    if filename.startswith("temp_"):
                        file_path = os.path.join(SPEAKER_FOLDER, filename)
                        if os.path.isfile(file_path):
                            try:
                                if os.stat(file_path).st_mtime < now - GC_TEMP_DAYS * 86400:
                                    os.remove(file_path)
                                    logger.info(f"GC deleted old temp speaker file: {filename}")
                            except Exception:
                                pass
                                
        except Exception as e:
            logger.error(f"Error during Garbage Collection: {e}")
            
        # Sleep for 24 hours (86400 seconds)
        await asyncio.sleep(86400)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(garbage_collector())

# Check for old format model version
XTTS.model_version = XTTS.check_model_version_old_format(MODEL_VERSION)
MODEL_VERSION = XTTS.model_version

# Create version string
version_string = ""
if MODEL_SOURCE == "api" or MODEL_VERSION == "main":
    version_string = "lastest"
else:
    version_string = MODEL_VERSION

# Load model
if STREAM_MODE or STREAM_MODE_IMPROVE:
    # Load model for Streaming
    check_stream2sentence_version()

    logger.warning("'Streaming Mode' has certain limitations, you can read about them here https://github.com/daswer123/xtts-api-server#about-streaming-mode")

    if STREAM_MODE_IMPROVE:
        logger.info("You launched an improved version of streaming, this version features an improved tokenizer and more context when processing sentences, which can be good for complex languages like Chinese")
        
    model_path = XTTS.model_folder
    
    engine = CoquiEngine(specific_model=MODEL_VERSION,use_deepspeed=DEEPSPEED,local_models_path=str(model_path))
    stream = TextToAudioStream(engine)
else:
  logger.info(f"Model: '{version_string}' starts to load,wait until it loads")
  XTTS.load_model() 

if USE_CACHE:
    logger.info("You have enabled caching, this option enables caching of results, your results will be saved and if there is a repeat request, you will get a file instead of generation")

# Add CORS middleware 
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Help funcs
def play_stream(stream,language):
  if STREAM_MODE_IMPROVE:
    # Here we define common arguments in a dictionary for DRY principle
    play_args = {
        'minimum_sentence_length': 2,
        'minimum_first_fragment_length': 2,
        'tokenizer': "stanza",
        'language': language,
        'context_size': 2
    }
    if STREAM_PLAY_SYNC:
        # Play synchronously
        stream.play(**play_args)
    else:
        # Play asynchronously
        stream.play_async(**play_args)
  else:
    # If not improve mode just call the appropriate method based on sync_play flag.
    if STREAM_PLAY_SYNC:
      stream.play()
    else:
      stream.play_async()

class OutputFolderRequest(BaseModel):
    output_folder: str

class SpeakerFolderRequest(BaseModel):
    speaker_folder: str

class ModelNameRequest(BaseModel):
    model_name: str

class TTSSettingsRequest(BaseModel):
    stream_chunk_size: int
    temperature: float
    speed: float
    length_penalty: float
    repetition_penalty: float
    top_p: float
    top_k: int
    enable_text_splitting: bool

class SynthesisRequest(BaseModel):
    text: str
    speaker_wav: str = "default_voice"
    language: str

class SynthesisFileRequest(BaseModel):
    text: str
    speaker_wav: str = "default_voice"
    language: str
    file_name_or_path: str  

@app.get("/speakers_list")
def get_speakers():
    speakers = XTTS.get_speakers()
    return speakers

@app.get("/speakers")
def get_speakers():
    speakers = XTTS.get_speakers_special()
    return speakers

@app.get("/languages")
def get_languages():
    languages = XTTS.list_languages()
    return {"languages": languages}

@app.get("/get_folders")
def get_folders():
    speaker_folder = XTTS.speaker_folder
    output_folder = XTTS.output_folder
    model_folder = XTTS.model_folder
    return {"speaker_folder": speaker_folder, "output_folder": output_folder,"model_folder":model_folder}

@app.get("/get_models_list")
def get_models_list():
    return XTTS.get_models_list()

@app.get("/get_tts_settings")
def get_tts_settings():
    settings = {**XTTS.tts_settings,"stream_chunk_size":XTTS.stream_chunk_size}
    return settings

@app.get("/sample/{file_name:path}")
def get_sample(file_name: str):
    # A fix for path traversal vulenerability. 
    # An attacker may summon this endpoint with ../../etc/passwd and recover the password file of your PC (in linux) or access any other file on the PC
    if ".." in file_name:
        raise HTTPException(status_code=404, detail=".. in the file name! Are you kidding me?") 
    file_path = os.path.join(XTTS.speaker_folder, file_name)
    if os.path.isfile(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    else:
        logger.error("File not found")
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/output/{file_name:path}")
def get_output(file_name: str):
    if ".." in file_name:
        raise HTTPException(status_code=404, detail=".. in the file name! Are you kidding me?") 
    file_path = os.path.join(XTTS.output_folder, file_name)
    if os.path.isfile(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    else:
        logger.error("File not found")
        raise HTTPException(status_code=404, detail="File not found")

@app.post("/set_output")
def set_output(output_req: OutputFolderRequest):
    try:
        XTTS.set_out_folder(output_req.output_folder)
        return {"message": f"Output folder set to {output_req.output_folder}"}
    except ValueError as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/set_speaker_folder")
def set_speaker_folder(speaker_req: SpeakerFolderRequest):
    try:
        XTTS.set_speaker_folder(speaker_req.speaker_folder)
        return {"message": f"Speaker folder set to {speaker_req.speaker_folder}"}
    except ValueError as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/switch_model")
def switch_model(modelReq: ModelNameRequest):
    try:
        XTTS.switch_model(modelReq.model_name)
        return {"message": f"Model switched to {modelReq.model_name}"}
    except InvalidSettingsError as e:  
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/set_tts_settings")
def set_tts_settings_endpoint(tts_settings_req: TTSSettingsRequest):
    try:
        XTTS.set_tts_settings(**tts_settings_req.dict())
        return {"message": "Settings successfully applied"}
    except InvalidSettingsError as e: 
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/tts_with_progress")
async def tts_with_progress(
    request: Request,
    text: str = Form(...),
    speaker_name: str = Form(None),
    speaker_wav: str = Form(None),
    speaker_file: Optional[UploadFile] = File(None),
    language: str = Form("auto")
):
    import json
    import asyncio
    import re
    import shutil
    from uuid import uuid4
    from fastapi.responses import StreamingResponse

    if not speaker_name and not speaker_wav and not speaker_file:
        speaker_name = "default_voice"

    temp_speaker_path = None
    if speaker_file is not None:
        temp_speaker_path = os.path.abspath(os.path.join(XTTS.speaker_folder, f"temp_{uuid4()}_{speaker_file.filename}"))
        try:
            with open(temp_speaker_path, "wb") as buffer:
                shutil.copyfileobj(speaker_file.file, buffer)
            XTTS.preprocess_speaker_audio(temp_speaker_path)
            speaker_wav = temp_speaker_path
            speaker_name = temp_speaker_path
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save speaker file: {str(e)}")
    elif not speaker_wav:
        speaker_wav = speaker_name

    if language.lower() == "auto":
        language = detect_language(text)

    if XTTS.model_source != "local":
        raise HTTPException(status_code=400, detail="Only local models are supported for this endpoint.")
    
    if language.lower() not in supported_languages:
        raise HTTPException(status_code=400, detail="Unsupported language.")

    async def event_generator():
        try:
            clear_text = XTTS.clean_text(text, language)
            
            # Tách câu cơ bản dựa trên dấu chấm, phẩy, hỏi chấm, than ôi
            sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', clear_text) if s.strip()]
            if not sentences:
                sentences = [clear_text]
                
            total = len(sentences)
            all_wavs = []
            output_file = os.path.join(XTTS.output_folder, f"progress_{uuid4()}.wav")
            
            for i, sentence in enumerate(sentences):
                # Gửi trạng thái % trước khi render
                progress = int((i / total) * 100)
                yield f'data: {json.dumps({"progress": progress, "status": f"rendering {i+1}/{total}"})}\n\n'
                
                # Render câu hiện tại trên threadpool
                wav_tensor = await asyncio.to_thread(
                    XTTS.generate_audio_tensor,
                    sentence,
                    speaker_name,
                    speaker_wav,
                    language
                )
                if wav_tensor is not None:
                    all_wavs.append(wav_tensor)
            
            # Gộp và lưu file
            if all_wavs:
                import torch
                import torchaudio
                final_wav = torch.cat(all_wavs, dim=1) 
                out_sample_rate = 48000 if XTTS.output_sample_rate_48k else 24000
                torchaudio.save(output_file, final_wav, out_sample_rate)
                
                filename = os.path.basename(output_file)
                # Ensure the path returned matches how files are served
                yield f'data: {json.dumps({"progress": 100, "status": "completed", "audio_url": f"/output/{filename}"})}\n\n'
            else:
                yield f'data: {json.dumps({"error": "Failed to generate audio"})}\n\n'
                
        except Exception as e:
            yield f'data: {json.dumps({"error": str(e)})}\n\n'
        finally:
            # Dọn dẹp file tạm và cache sau khi hoàn thành stream
            if temp_speaker_path and os.path.exists(temp_speaker_path):
                try:
                    os.remove(temp_speaker_path)
                    if temp_speaker_path in XTTS.latents_cache:
                        del XTTS.latents_cache[temp_speaker_path]
                except Exception as e:
                    logger.error(f"Failed to cleanup temp speaker file: {e}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get('/tts_stream')
async def tts_stream(request: Request, text: str = Query(), speaker_wav: str = Query(default="default_voice"), language: str = Query()):
    if not speaker_wav:
        speaker_wav = "default_voice"

    if language.lower() == "auto":
        language = detect_language(text)

    # Validate local model source.
    if XTTS.model_source != "local":
        raise HTTPException(status_code=400,
                            detail="HTTP Streaming is only supported for local models.")
    # Validate language code against supported languages.
    if language.lower() not in supported_languages and language.lower() != "auto":
        raise HTTPException(status_code=400,
                            detail="Language code sent is either unsupported or misspelled.")
            
    async def generator():
        global queue_count
        if queue_count >= MAX_QUEUE_SIZE:
            raise HTTPException(status_code=429, detail="Server is overloaded. Too many requests in queue.")
        
        queue_count += 1
        try:
            async with tts_queue:
                chunks = XTTS.process_tts_to_file(
                    text=text,
                    speaker_name_or_path=speaker_wav,
                    language=language.lower(),
                    stream=True,
                )
                # Write file header to the output stream.
                yield XTTS.get_wav_header()
                async for chunk in chunks:
                    # Check if the client is still connected.
                    disconnected = await request.is_disconnected()
                    if disconnected:
                        break
                    yield chunk
        finally:
            queue_count -= 1

    return StreamingResponse(generator(), media_type='audio/x-wav')

@app.post("/tts_to_audio/")
async def tts_to_audio(
    background_tasks: BackgroundTasks,
    text_file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    speaker_wav: Optional[UploadFile] = File(None),
    speaker_name: Optional[str] = Form(None),
    language: str = Form("vi")
):
    final_text = ""
    if text_file is not None:
        try:
            content = await text_file.read()
            try:
                final_text = content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    final_text = content.decode('utf-16')
                except UnicodeDecodeError:
                    final_text = content.decode('utf-8-sig', errors='ignore')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read text file: {str(e)}")
    elif text is not None:
        final_text = text
    else:
        raise HTTPException(status_code=400, detail="Either text_file or text must be provided.")
        
    if not final_text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")

    if language.lower() == "auto":
        language = detect_language(final_text)

    temp_speaker_path = None
    final_speaker = None
    
    if speaker_wav is not None:
        temp_speaker_path = os.path.abspath(os.path.join(XTTS.speaker_folder, f"temp_{uuid4()}_{speaker_wav.filename}"))
        try:
            with open(temp_speaker_path, "wb") as buffer:
                import shutil
                shutil.copyfileobj(speaker_wav.file, buffer)
            XTTS.preprocess_speaker_audio(temp_speaker_path)
            final_speaker = temp_speaker_path
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save speaker file: {str(e)}")
    elif speaker_name is not None:
        final_speaker = speaker_name
    else:
        final_speaker = "default_voice"

    if STREAM_MODE or STREAM_MODE_IMPROVE:
        try:
            global stream
            # Validate language code against supported languages.
            if language.lower() not in supported_languages and language.lower() != "auto":
                raise HTTPException(status_code=400,
                                    detail="Language code sent is either unsupported or misspelled.")

            speaker_wav_path = XTTS.get_speaker_wav(final_speaker)
            lang_code = language[0:2]

            if stream.is_playing() and not STREAM_PLAY_SYNC:
                stream.stop()
                stream = TextToAudioStream(engine)

            engine.set_voice(speaker_wav_path)
            engine.language = language.lower()
           
            # Start streaming, works only on your local computer.
            stream.feed(final_text)
            play_stream(stream,lang_code)

            # It's a hack, just send 1 second of silence so that there is no sillyTavern error.
            this_dir = Path(__file__).parent.resolve()
            output = this_dir / "RealtimeTTS" / "silence.wav"

            return FileResponse(
                path=output,
                media_type='audio/wav',
                filename="silence.wav",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    else:
        try:
            if XTTS.model_source == "local":
              logger.info(f"Processing TTS to audio with text length: {len(final_text)}")

            # Validate language code against supported languages.
            if language.lower() not in supported_languages and language.lower() != "auto":
                raise HTTPException(status_code=400,
                                    detail="Language code sent is either unsupported or misspelled.")

            # Generate an audio file using process_tts_to_file.
            global queue_count
            if queue_count >= MAX_QUEUE_SIZE:
                raise HTTPException(status_code=429, detail="Server is overloaded. Too many requests in queue.")
            
            queue_count += 1
            try:
                async with tts_queue:
                    output_file_path = await run_in_threadpool(
                        XTTS.process_tts_to_file,
                        text=final_text,
                        speaker_name_or_path=final_speaker,
                        language=language.lower(),
                        file_name_or_path=f'{str(uuid4())}.wav'
                    )
            finally:
                queue_count -= 1

            def cleanup(speaker_path, out_path, enable_cache):
                if speaker_path and os.path.exists(speaker_path):
                    os.remove(speaker_path)
                    if speaker_path in XTTS.latents_cache:
                        del XTTS.latents_cache[speaker_path]
                if not enable_cache and out_path and os.path.exists(out_path):
                    os.remove(out_path)

            background_tasks.add_task(cleanup, temp_speaker_path, output_file_path, XTTS.enable_cache_results)

            # Return the file in the response
            return FileResponse(
                path=output_file_path,
                media_type='audio/wav',
                filename="output.wav",
                )

        except Exception as e:
            if temp_speaker_path and os.path.exists(temp_speaker_path):
                os.remove(temp_speaker_path)
            logger.error(e)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/tts_to_file")
async def tts_to_file(request: SynthesisFileRequest):
    try:
        if not request.speaker_wav:
            request.speaker_wav = "default_voice"

        if XTTS.model_source == "local":
          logger.info(f"Processing TTS to file with request: {request}")

        if request.language.lower() == "auto":
            request.language = detect_language(request.text)

        # Validate language code against supported languages.
        if request.language.lower() not in supported_languages and request.language.lower() != "auto":
             raise HTTPException(status_code=400,
                                 detail="Language code sent is either unsupported or misspelled.")

        # Now use process_tts_to_file for saving the file.
        global queue_count
        if queue_count >= MAX_QUEUE_SIZE:
            raise HTTPException(status_code=429, detail="Server is overloaded. Too many requests in queue.")
        
        queue_count += 1
        try:
            async with tts_queue:
                output_file = await run_in_threadpool(
                    XTTS.process_tts_to_file,
                    text=request.text,
                    speaker_name_or_path=request.speaker_wav,
                    language=request.language.lower(),
                    file_name_or_path=request.file_name_or_path  # The user-provided path to save the file is used here.
                )
        finally:
            queue_count -= 1
        return {"message": "The audio was successfully made and stored.", "output_path": output_file}

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/tts_from_files/")
async def tts_from_files(
    background_tasks: BackgroundTasks,
    language: str = Form(...),
    speaker_file: Optional[UploadFile] = File(None),
    text_file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    import shutil
    if XTTS.model_source != "local":
        raise HTTPException(status_code=400, detail="Only local models are supported for this endpoint.")

    # 1. Read text from text_file or Form text
    final_text = ""
    if text_file is not None:
        try:
            content = await text_file.read()
            try:
                final_text = content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    final_text = content.decode('utf-16')
                except UnicodeDecodeError:
                    final_text = content.decode('utf-8-sig', errors='ignore')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read text file: {str(e)}")
    elif text is not None:
        final_text = text
    else:
        raise HTTPException(status_code=400, detail="Either text_file or text must be provided.")

    if not final_text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")

    if language.lower() == "auto":
        language = detect_language(final_text)
    elif language.lower() not in supported_languages:
        raise HTTPException(status_code=400, detail="Language code sent is either unsupported or misspelled.")

    # 2. Save speaker audio to a temporary file or use default
    temp_speaker_path = None
    if speaker_file is not None:
        temp_speaker_path = os.path.abspath(os.path.join(XTTS.speaker_folder, f"temp_{uuid4()}_{speaker_file.filename}"))
        try:
            with open(temp_speaker_path, "wb") as buffer:
                shutil.copyfileobj(speaker_file.file, buffer)
            XTTS.preprocess_speaker_audio(temp_speaker_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save speaker file: {str(e)}")
        final_speaker = temp_speaker_path
    else:
        final_speaker = "default_voice"

    # 3. Generate audio
    try:
        global queue_count
        if queue_count >= MAX_QUEUE_SIZE:
            raise HTTPException(status_code=429, detail="Server is overloaded. Too many requests in queue.")
        
        queue_count += 1
        try:
            async with tts_queue:
                output_file_path = await run_in_threadpool(
                    XTTS.process_tts_to_file,
                    text=final_text,
                    speaker_name_or_path=final_speaker,
                    language=language.lower(),
                    file_name_or_path=f'{str(uuid4())}.wav'
                )
        finally:
            queue_count -= 1

        # 4. Clean up temporary files
        def cleanup(speaker_path, output_path, enable_cache):
            if speaker_path and os.path.exists(speaker_path):
                os.remove(speaker_path)
            if not enable_cache and output_path and os.path.exists(output_path):
                os.remove(output_path)
            
            if speaker_path and speaker_path in XTTS.latents_cache:
                del XTTS.latents_cache[speaker_path]

        background_tasks.add_task(cleanup, temp_speaker_path, output_file_path, XTTS.enable_cache_results)

        return FileResponse(
            path=output_file_path,
            media_type='audio/wav',
            filename="output.wav"
        )
    except Exception as e:
        if temp_speaker_path and os.path.exists(temp_speaker_path):
            os.remove(temp_speaker_path)
        logger.error(e)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8002)
