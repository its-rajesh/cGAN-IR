import numpy as np
import os


def RoomImpulseResponse(source_signals, room_dim, delay_time):
    
    vocals, drums, other = source_signals[0], source_signals[1], source_signals[2]
    fs = 22050
    rt60_tgt = 0.25 #seconds reverberation
    while True:
        try:
            e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
            print(f"Energy absorption: {e_absorption}, Maximum order: {max_order}")
            break  # exit loop if successful
        except ValueError as e:
            print(f"{e} Trying with a longer RT60...")
            rt60_tgt += 0.1  # increment RT60 by 0.1 seconds

    print(f"Final RT60: {rt60_tgt}")
    
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

    # Microphone positions [x, y, z] in meters
    source_pos = np.array([[4, 5, 1], [7, 5, 1], [9, 5, 1]])

    # Sound source positions [x, y, z] in meters
    mic_pos = np.array([[4, 5.5, 1], [7, 5.5, 1], [9, 5.5, 1]]).T

    # Create a ShoeBox room
    #m = pra.Material(energy_absorption="hard_surface")
    #pra.Material(e_absorption)
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order)
    #room = pra.ShoeBox(room_dim, fs=fs)
    
    room.add_source(source_pos[0], signal=drums, delay=delay_time)
    room.add_source(source_pos[1], signal=vocals, delay=delay_time)
    room.add_source(source_pos[2], signal=other, delay=delay_time)
    
    # Add microphone array to the room
    mic_array = pra.MicrophoneArray(mic_pos, fs=fs)
    room.add_microphone_array(mic_array)
    
    room.simulate()
    room.compute_rir()

    result = []
    for i, mic in enumerate(mic_array.R.T):
        mic_signal = mic_array.signals[i]
        result.append(mic_signal)
    
    return result




path = "/home/anchal/Desktop/rajesh/Clean/Dataset/MUSDBHQ/LM/Ytrain.npy"

Ytrain = np.load(path)
print('Ytrain Shape:', Ytrain.shape)

dim = 110240

Ytrain_modified = np.delete(Ytrain, 2, axis=1)
Ytrain_split_combined = np.concatenate(np.array_split(Ytrain_modified, 2, axis=2), axis=0)
Ytrain_new = Ytrain_split_combined[:, :, :dim]
print('New Ytrain Shape:', Ytrain_new.shape)

Ytrain = Ytrain_new

rir_songs = []
for si in tqdm(Ytrain[:, :, :]):
    # Room dimensions [length, width, height] in meters
    room_dim = [np.random.randint(8, 15), np.random.randint(8, 15), np.random.randint(6, 12)]
    #room_dim = [10, 10, 8]
    #delay_time = np.random.randint(1, 5)*1e-3 #20-50ms
    delay_time = 1e-6
    same_song_outs = RoomImpulseResponse(si, room_dim, delay_time)
    v, d, o = same_song_outs[1], same_song_outs[0], same_song_outs[2]
    rir_songs.append([v[:dim], d[:dim], o[:dim]])
    print('Room dim:', room_dim)


np.save('/home/anchal/Desktop/rajesh/Clean/Dataset/MUSDBHQ/CM/XtrainCM3.npy', rir_songs)
np.save('/home/anchal/Desktop/rajesh/Clean/Dataset/MUSDBHQ/CM/YtrainCM3.npy', Ytrain)