"""Small repository-owned golden V1 graph used for compatibility testing."""

from __future__ import annotations

from junctionlens.contract.ids import edge_id, predicted_node_id
from junctionlens.v1 import scene_control_graph_pb2 as scg

_IDENTITY3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
_IDENTITY4 = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def _polyline(points: tuple[tuple[float, float, float], ...]) -> scg.Polyline3d:
    polyline = scg.Polyline3d(confidence=0.95)
    for x, y, z in points:
        polyline.points.add(x=x, y=y, z=z)
        polyline.point_uncertainty.add(x=0.1, y=0.1, z=0.1)
    return polyline


def make_golden_envelope() -> scg.SceneControlGraphEnvelope:
    """Construct a complete graph containing every principal V1 node kind."""
    envelope = scg.SceneControlGraphEnvelope(schema_major=1, schema_minor=0)
    envelope.producer.git_commit = "1" * 40
    envelope.producer.model_artifact_sha256 = "2" * 64
    envelope.producer.configuration_sha256 = "3" * 64
    envelope.producer.runtime_build_sha256 = "4" * 64
    envelope.producer.execution_provider_profile = "cpu-reference"
    envelope.producer.provider_assignment_digest = "5" * 64
    envelope.producer.random_seed = 7

    graph = envelope.graph
    graph.role = scg.GRAPH_ROLE_PREDICTION
    graph.frame_key.dataset_id = "synthetic"
    graph.frame_key.dataset_version = "v1"
    graph.frame_key.split_id = "golden"
    graph.frame_key.segment_id = "segment-0001"
    graph.frame_key.timestamp_ns = 1_725_000_000_000_000_000
    graph.frame_key.source_domain = scg.SOURCE_DOMAIN_SYNTHETIC
    graph.frame_key.calibration_sha256 = "6" * 64
    graph.frame_key.frame_manifest_sha256 = "7" * 64

    sensor = graph.sensor_frame
    sensor.frame_key.CopyFrom(graph.frame_key)
    sensor.t_world_vehicle.values.extend(_IDENTITY4)
    sensor.pose_valid = True
    sensor.adapter_version = "golden-v1"
    for slot in range(scg.CAMERA_SLOT_FRONT_CENTER, scg.CAMERA_SLOT_REAR_RIGHT + 1):
        camera = sensor.cameras.add(slot=slot, valid=slot == scg.CAMERA_SLOT_FRONT_CENTER)
        camera.capture_timestamp_ns = graph.frame_key.timestamp_ns
        camera.original_width = 1920 if camera.valid else 0
        camera.original_height = 1080 if camera.valid else 0
        camera.intrinsic.values.extend((1000.0, 0.0, 960.0, 0.0, 1000.0, 540.0, 0.0, 0.0, 1.0))
        camera.t_vehicle_camera.values.extend(_IDENTITY4)
        camera.distortion_model = scg.DISTORTION_MODEL_NONE
        camera.image_transform.original_to_model.values.extend(_IDENTITY3)
        camera.image_transform.resized_width = 1920
        camera.image_transform.resized_height = 1080
        if camera.valid:
            camera.original_image.kind = scg.ARTIFACT_KIND_SOURCE_IMAGE
            camera.original_image.sha256 = "8" * 64
            camera.original_image.byte_size = 12345
            camera.original_image.media_type = "image/jpeg"
            camera.original_image.relative_uri = "images/front-center.jpg"
            camera.original_image.license_id = "LicenseRef-JunctionLens-Synthetic"

    lane_id = predicted_node_id(scg.NODE_TYPE_LANE_SEGMENT, 0)
    control_id = predicted_node_id(scg.NODE_TYPE_TRAFFIC_CONTROL, 0)
    area_id = predicted_node_id(scg.NODE_TYPE_ROAD_AREA, 0)
    lane = graph.lanes.add(node_id=lane_id, track_id=101, decoder_query_index=2)
    lane.centerline.CopyFrom(_polyline(((0.0, 0.0, 0.0), (0.0, 10.0, 0.0))))
    lane.left_boundary.CopyFrom(_polyline(((-1.5, 0.0, 0.0), (-1.5, 10.0, 0.0))))
    lane.right_boundary.CopyFrom(_polyline(((1.5, 0.0, 0.0), (1.5, 10.0, 0.0))))
    lane.left_boundary_type.probabilities.extend((0.1, 0.9))
    lane.right_boundary_type.probabilities.extend((0.8, 0.2))
    lane.intersection_or_connector_probability = 0.2
    lane.existence_confidence = 0.99
    for _ in lane.centerline.points:
        lane.centerline_laplace_scale_m.add(x=0.1, y=0.1, z=0.1)

    control = graph.traffic_controls.add(
        node_id=control_id,
        track_id=102,
        source_camera=scg.CAMERA_SLOT_FRONT_CENTER,
        existence_confidence=0.98,
        calibrated_class_confidence=0.96,
        calibrated_attribute_confidence=0.92,
        decoder_query_index=4,
    )
    control.source_pixel_box.x0 = 100.25
    control.source_pixel_box.y0 = 200.5
    control.source_pixel_box.x1 = 180.75
    control.source_pixel_box.y1 = 320.25
    control.source_pixel_box.convention = scg.SOURCE_BOX_CONVENTION_XYXY_HALF_OPEN
    control.source_pixel_box.image_width = 1920
    control.source_pixel_box.image_height = 1080
    control.normalized_half_open_box.x_min = 100.25 / 1920.0
    control.normalized_half_open_box.y_min = 200.5 / 1080.0
    control.normalized_half_open_box.x_max = 180.75 / 1920.0
    control.normalized_half_open_box.y_max = 320.25 / 1080.0
    control.category_distribution.probabilities.extend((0.05, 0.9, 0.05))
    control.attribute_distribution.probabilities.extend((0.1, 0.2, 0.7))

    area = graph.road_areas.add(node_id=area_id, track_id=103, existence_confidence=0.9)
    area.category_distribution.probabilities.extend((0.85, 0.15))
    area.geometry.CopyFrom(
        _polyline(((-3.0, 5.0, 0.0), (3.0, 5.0, 0.0), (3.0, 8.0, 0.0), (-3.0, 8.0, 0.0)))
    )
    for _ in area.geometry.points:
        area.geometry_uncertainty.add(x=0.2, y=0.2, z=0.2)

    edge = graph.edges.add(
        edge_type=scg.GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE,
        source_node_id=control_id,
        target_node_id=lane_id,
        raw_probability=0.91,
        calibrated_probability=0.88,
        binary_decision=True,
    )
    edge.edge_id = edge_id(
        graph.frame_key, edge.edge_type, edge.source_node_id, edge.target_node_id
    )
    edge.uncertainty.standard_deviation = 0.03
    edge.uncertainty.method = "bootstrap"

    for track_id, node_type, node_id in (
        (101, scg.NODE_TYPE_LANE_SEGMENT, lane_id),
        (102, scg.NODE_TYPE_TRAFFIC_CONTROL, control_id),
        (103, scg.NODE_TYPE_ROAD_AREA, area_id),
    ):
        track = graph.tracks.add(
            track_id=track_id,
            node_type=node_type,
            current_node_id=node_id,
            first_timestamp_ns=graph.frame_key.timestamp_ns,
            last_timestamp_ns=graph.frame_key.timestamp_ns,
            age_observed_frames=1,
            termination_reason=scg.TRACK_TERMINATION_REASON_ACTIVE,
        )
        track.matching_cost.geometry = 0.1
        track.matching_cost.class_cost = 0.2
        track.matching_cost.motion = 0.05
        track.matching_cost.total = 0.35
    return envelope
