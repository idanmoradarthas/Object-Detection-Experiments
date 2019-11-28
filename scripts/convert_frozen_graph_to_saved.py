from pathlib import Path

import tensorflow
from tensorflow.python.saved_model import signature_constants, tag_constants

if __name__ == '__main__':
    version = 1
    model_name = "faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28"
    export_dir = str(
        Path(__file__).parent.joinpath("saved_model").joinpath(model_name).joinpath(f"{version}").absolute())
    path_to_frozen_graph = Path("__file__").parent.absolute().parent.joinpath("downloaded_models").joinpath(
        model_name).joinpath("frozen_inference_graph.pb")

    builder = tensorflow.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
    print("created output folder")

    detection_graph = tensorflow.Graph()
    with detection_graph.as_default():
        od_graph_def = tensorflow.compat.v1.GraphDef()
        with tensorflow.io.gfile.GFile(str(path_to_frozen_graph.absolute()), "rb") as g_file:
            serialized_graph = g_file.read()
            od_graph_def.ParseFromString(serialized_graph)
            tensorflow.import_graph_def(od_graph_def, name="")
    print("fininsh loading graph")

    with tensorflow.compat.v1.Session(graph=detection_graph) as sess:
        input = detection_graph.get_tensor_by_name('image_tensor:0')

        tensor_dict = {}
        ops = detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = detection_graph.get_tensor_by_name(tensor_name)

        sigs = {
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tensorflow.compat.v1.saved_model.signature_def_utils.predict_signature_def(
                {"in": input}, tensor_dict)}
        builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], signature_def_map=sigs)

    print("Saving Model")
    builder.save()
    print('Export SavedModel!')
