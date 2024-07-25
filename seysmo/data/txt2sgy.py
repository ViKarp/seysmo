import pandas as pd
import numpy as np
import segyio


def txt2sgy(input_filepath: str, output_filepath: str) -> None:
    df = pd.read_csv(input_filepath, sep=",", index_col=None)
    grouped_df = df.groupby(['Receiver Midpoint_X', 'Receiver Midpoint_Y'])
    max_samples = max(group.shape[0] for name, group in grouped_df)
    spec = segyio.spec()
    spec.ilines = np.arange(1, len(grouped_df)+1)
    spec.xlines = np.arange(1)
    spec.samples = np.arange(max_samples)
    spec.format = 1

    with segyio.create(output_filepath, spec) as f:
        f.text[0] = segyio.tools.wrap("Created with segyio")
        f.bin.update({
            segyio.BinField.Interval: 100,
            segyio.BinField.Samples: max_samples,
            segyio.BinField.Format: 1,
        })

        trace_idx = 0

        for (x, y), group in grouped_df:
            group = group.sort_values(by='Depth', ascending=False)

            trace_data = np.zeros(max_samples, dtype=np.float32)
            trace_data[:len(group['Velocity'])] = group['Velocity'].values

            f.trace[trace_idx] = trace_data

            f.header[trace_idx].update({
                segyio.TraceField.SourceGroupScalar: -100,
                segyio.TraceField.ElevationScalar: -100,
                segyio.TraceField.CDP_X: int(x*100),
                segyio.TraceField.CDP_Y: int(y*100),
                segyio.TraceField.SourceX: int(x * 100),
                segyio.TraceField.SourceY: int(y * 100),
                segyio.TraceField.TRACE_SAMPLE_COUNT: max_samples,
                segyio.TraceField.TRACE_SAMPLE_INTERVAL: 10000,

            })

            trace_idx += 1


if __name__ == "__main__":
    txt2sgy('../../data/interim/OUTPUT/Profile.txt', '../../data/interim/OUTPUT/Profile.sgy')
