import re
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

STATS_VALUE_REGEX = re.compile(r'^(\d+)\((\d+)/(\d+)/(\d+)\)')


# ---------------------------------------------------------------------------
# Hex-packed binary TLV decoder for newer firmwares (paramVersion >= ~16).
#
# Reverse-engineered from the RainPoint Home APK
# (com.baldr.homgar.bean.DpDeviceStatus.analyzeDpDeviceStatus). The value
# looks like "<firmware>#<hex bytes>" with no leading general-status ';'.
# ---------------------------------------------------------------------------

@dataclass
class DpRecord:
    """One TLV record from a Dxx status value."""
    dp_id: int          # e.g. 25 = STA_WKSTATE port 1 on HTV213FRF/model-288
    type_code: int      # the dpCode from the productModel catalog
    type_len: int       # number of payload data bytes (excluding header)
    payload: bytes      # the payload bytes proper


def parse_tlv_d_value(value: str) -> List[DpRecord]:
    """Parse a single ``Dxx`` status value in the hex-packed TLV format.

    The string ``"11#17E1C2..."`` is composed of an ASCII firmware-version
    prefix (``11``), a ``#`` separator and the binary payload encoded as hex.
    Each record is 1 byte of ``dpId`` followed by either:

    * SHORT form (top bit clear): 1 byte header whose bits 6..4 are the
      ``typeCode`` and whose bits 3..0 are the sole data nibble.
    * LONG form (top bit set):
        ``i15 = (h >> 2) & 0x1F`` (extended when 31),
        ``i16 = h & 3`` → ``typeLen = i16 + 1``,
        typeCode = ``i15 + 8`` (or ``buf[next] + 39`` when extended),
        payload spans ``typeLen`` bytes after the header byte.

    Anything after the first ``,`` is ignored — that's a legacy hook the
    older format used.
    """
    comma = value.find(",")
    if comma != -1:
        value = value[:comma]
    if "#" in value:
        # Strip "<2 chars>#", e.g. "11#".
        value = value[3:]
    buf = bytes.fromhex(value)

    out: List[DpRecord] = []
    i = 0
    n = len(buf)
    while i < n:
        dp_id = buf[i]
        i += 1
        if i >= n:
            break
        header = buf[i]
        if (header >> 7) & 1 == 0:
            # SHORT: single byte, payload is the low nibble.
            type_code = (header >> 4) & 7
            payload = bytes([header & 0x0F])
            type_len = 1
            i += 1
        else:
            i15 = (header >> 2) & 0x1F
            i16 = header & 3
            type_len = i16 + 1
            span = i16 + 2  # header + type_len bytes (overlapping on some)
            if i15 <= 30:
                type_code = i15 + 8
                payload = buf[i + 1:i + span]
                i += span
            else:
                # extended code — next byte is the type code offset
                i += 1
                if i >= n:
                    break
                type_code = buf[i] + 39
                payload = buf[i + 1:i + span]
                i += span
        out.append(DpRecord(dp_id=dp_id, type_code=type_code,
                             type_len=type_len, payload=payload))
    return out


def _to_int_le(payload: bytes, signed: bool = False) -> int:
    if not payload:
        return 0
    n = len(payload)
    if signed:
        if n == 1:
            return struct.unpack("<b", payload[:1])[0]
        if n == 2:
            return struct.unpack("<h", payload[:2])[0]
        if n == 4:
            return struct.unpack("<i", payload[:4])[0]
        return int.from_bytes(payload, "little", signed=True)
    if n == 1:
        return payload[0]
    if n == 2:
        return struct.unpack("<H", payload[:2])[0]
    if n == 4:
        return struct.unpack("<I", payload[:4])[0]
    return int.from_bytes(payload, "little")


def _parse_stats_value(s):
    if match := STATS_VALUE_REGEX.fullmatch(s):
        return int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
    else:
        return None, None, None, None


def _temp_to_mk(f):
    return round(1000 * ((int(f) * .1 - 32) * 5 / 9 + 273.15))


class HomgarHome:
    """
    Represents a home in Homgar.
    A home can have a number of hubs, each of which can contain sensors/controllers (subdevices).
    """
    def __init__(self, hid, name):
        self.hid = hid
        self.name = name


class HomgarDevice:
    """
    Base class for Homgar devices; both hubs and subdevices.
    Each device has a model (name and code), name, some identifiers and may have alerts.
    """

    FRIENDLY_DESC = "Unknown HomGar device"

    def __init__(self, model, model_code, name, did, mid, alerts,
                 device_name=None, product_key=None, iot_id=None, sid=None,
                 **kwargs):
        self.model = model
        self.model_code = model_code
        self.name = name
        self.did = did  # the unique device identifier of this device itself
        self.mid = mid  # the unique identifier of the sensor network
        self.alerts = alerts

        # Identifiers needed for the control endpoint. Only hubs populate
        # device_name/product_key but we accept them on all classes so they
        # flow through kwargs cleanly from get_devices_for_hid.
        self.device_name = device_name
        self.product_key = product_key
        self.iot_id = iot_id
        self.sid = sid

        self.address = None
        self.rf_rssi = None

    def __str__(self):
        return f"{self.FRIENDLY_DESC} \"{self.name}\" (DID {self.did})"

    def get_device_status_ids(self) -> List[str]:
        """
        The response for /app/device/getDeviceStatus contains a subDeviceStatus for each of the subdevices.
        This function returns which IDs in the subDeviceStatus apply to this device.
        Usually this is just Dxx where xx is the device address, but the hub has some additional special keys.
        set_device_status() will be called on this object for all subDeviceStatus entries matching any of the
        return IDs.
        :return: The subDeviceStatus this device should listen to.
        """
        return []

    def set_device_status(self, api_obj: dict) -> None:
        """
        Called after a call to /app/device/getDeviceStatus with an entry from $.data.subDeviceStatus
        that matches one of the IDs returned by get_device_status_ids().
        Should update the device status with the contents of the given API response.
        :param api_obj: The $.data.subDeviceStatus API response that should be used to update this device's status
        """
        if api_obj['id'] == f"D{self.address:02d}":
            self._parse_status_d_value(api_obj['value'])

    def _parse_status_d_value(self, val: str) -> None:
        """
        Parses a $.data.subDeviceStatus[x].value field for an entry with ID 'Dxx' where xx is the device address.

        Two on-wire formats in the wild:

        * Legacy (paramVersion<=2): ``<general>;<specific>`` where ``<general>``
          is ``flag,rssi,flag`` (per _parse_general_status_d_value).
        * Newer (paramVersion>=16): no ``;`` — the whole value is the
          device-specific blob (typically the hex-TLV format handled by
          classes like ``RainPoint2ZoneTimer_V2``).

        The parse is wrapped in a try/except so one mis-formatted sub-device
        doesn't bring down the whole poll — status fields the subclass can't
        decode are simply ignored.

        :param val: Value of the $.data.subDeviceStatus[x].value field to apply
        """
        try:
            if ';' in val:
                general_str, specific_str = val.split(';', 1)
                self._parse_general_status_d_value(general_str)
            else:
                specific_str = val
            self._parse_device_specific_status_d_value(specific_str)
        except Exception as e:  # noqa: BLE001 — best-effort per sub-device
            import logging
            logging.getLogger(__name__).debug(
                "%s: failed to parse status value %r: %s",
                type(self).__name__, val, e,
            )

    def _parse_general_status_d_value(self, s: str):
        """
        Parses the part of a $.data.subDeviceStatus[x].value field before the ';' character,
        which has the same format for all subdevices. It has three ','-separated fields. The first and last fields
        are always '1' in my case, I presume it's to do with battery state / connection state.
        The second field is the RSSI in dBm.
        :param s: The value to parse and apply
        """
        unknown_1, rf_rssi, unknown_2 = s.split(',')
        self.rf_rssi = int(rf_rssi)

    def _parse_device_specific_status_d_value(self, s: str):
        """
        Parses the part of a $.data.subDeviceStatus[x].value field after the ';' character,
        which is in a device-specific format.
        Should update the device state.
        :param s: The value to parse and apply
        """
        raise NotImplementedError()


class HomgarHubDevice(HomgarDevice):
    """
    A hub acts as a gateway for sensors and actuators (subdevices).
    A home contains an arbitrary number of hubs, each of which contains an arbitrary number of subdevices.
    """
    def __init__(self, subdevices, **kwargs):
        super().__init__(**kwargs)
        self.address = 1
        self.subdevices = subdevices

    def __str__(self):
        return f"{super().__str__()} with {len(self.subdevices)} subdevices"

    def _parse_device_specific_status_d_value(self, s):
        pass


class HomgarSubDevice(HomgarDevice):
    """
    A subdevice is a device that is associated with a hub.
    It can be a sensor or an actuator.
    """
    def __init__(self, address, port_number, port_describe=None, **kwargs):
        super().__init__(**kwargs)
        self.address = address  # device address within the sensor network
        self.port_number = port_number  # the number of ports on the device, e.g. 2 for the 2-zone water timer
        # Pipe-separated per-port labels from /app/device/getDeviceByHid's
        # `portDescribe` field (e.g. "Sprinklers|Dripline"). Split lazily.
        self.port_describe_raw = port_describe or ""

    def port_label(self, port: int) -> str:
        """Return the user-set label for a port (1-based), or 'Port N'."""
        labels = [p for p in self.port_describe_raw.split("|") if p]
        if 1 <= port <= len(labels):
            return labels[port - 1]
        return f"Port {port}"

    def __str__(self):
        return f"{super().__str__()} at address {self.address}"

    def get_device_status_ids(self):
        return [f"D{self.address:02d}"]

    def _parse_device_specific_status_d_value(self, s):
        pass


class RainPointDisplayHub(HomgarHubDevice):
    MODEL_CODES = [264]
    FRIENDLY_DESC = "Irrigation Display Hub"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wifi_rssi = None
        self.battery_state = None
        self.connected = None

        self.temp_mk_current = None
        self.temp_mk_daily_max = None
        self.temp_mk_daily_min = None
        self.temp_trend = None
        self.hum_current = None
        self.hum_daily_max = None
        self.hum_daily_min = None
        self.hum_trend = None
        self.press_pa_current = None
        self.press_pa_daily_max = None
        self.press_pa_daily_min = None
        self.press_trend = None

    def get_device_status_ids(self):
        return ["connected", "state", "D01"]

    def set_device_status(self, api_obj):
        dev_id = api_obj['id']
        val = api_obj['value']
        if dev_id == "state":
            self.battery_state, self.wifi_rssi = [int(s) for s in val.split(',')]
        elif dev_id == "connected":
            self.connected = int(val) == 1
        else:
            super().set_device_status(api_obj)

    def _parse_device_specific_status_d_value(self, s):
        """
        Observed example value:
        781(781/723/1),52(64/50/1),P=10213(10222/10205/1),

        Deduced meaning:
        temp[.1F](day-max/day-min/trend?),humidity[%](day-max/day-min/trend?),P=pressure[Pa](day-max/day-min/trend?),
        """
        temp_str, hum_str, press_str, *_ = s.split(',')
        self.temp_mk_current, self.temp_mk_daily_max, self.temp_mk_daily_min, self.temp_trend = [_temp_to_mk(v) for v in _parse_stats_value(temp_str)]
        self.hum_current, self.hum_daily_max, self.hum_daily_min, self.hum_trend = _parse_stats_value(hum_str)
        self.press_pa_current, self.press_pa_daily_max, self.press_pa_daily_min, self.press_trend = _parse_stats_value(press_str[2:])

    def __str__(self):
        s = super().__str__()
        if self.temp_mk_current:
            s += f": {self.temp_mk_current*1e-3:.1f}K / {self.hum_current}% / {self.press_pa_current}Pa"
        return s


class RainPointSoilMoistureSensor(HomgarSubDevice):
    MODEL_CODES = [72]
    FRIENDLY_DESC = "Soil Moisture Sensor"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.temp_mk_current = None
        self.moist_percent_current = None
        self.light_lux_current = None

    def _parse_device_specific_status_d_value(self, s):
        """
        Observed example value:
        766,52,G=31351

        Deduced meaning:
        temp[.1F],soil-moisture[%],G=light[.1lux]
        """
        temp_str, moist_str, light_str = s.split(',')
        self.temp_mk_current = _temp_to_mk(temp_str)
        self.moist_percent_current = int(moist_str)
        self.light_lux_current = int(light_str[2:]) * .1

    def __str__(self):
        s = super().__str__()
        if self.temp_mk_current:
            s += f": {self.temp_mk_current*1e-3-273.15:.1f}°C / {self.moist_percent_current}% / {self.light_lux_current:.1f}lx"
        return s


class RainPointRainSensor(HomgarSubDevice):
    MODEL_CODES = [87]
    FRIENDLY_DESC = "High Precision Rain Sensor"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rainfall_mm_total = None
        self.rainfall_mm_hour = None
        self.rainfall_mm_daily = None
        self.rainfall_mm_total = None

    def _parse_device_specific_status_d_value(self, s):
        """
        Observed example value:
        R=270(0/0/270)

        Deduced meaning:
        R=total?[.1mm](hour?[.1mm]/24hours?[.1mm]/7days?[.1mm])
        """
        self.rainfall_mm_total, self.rainfall_mm_hour, self.rainfall_mm_daily, self.rainfall_mm_7days = [.1*v for v in _parse_stats_value(s[2:])]

    def __str__(self):
        s = super().__str__()
        if self.rainfall_mm_total:
            s += f": {self.rainfall_mm_total}mm total / {self.rainfall_mm_hour}mm 1h / {self.rainfall_mm_daily}mm 24h / {self.rainfall_mm_7days}mm 7days"
        return s


class RainPointAirSensor(HomgarSubDevice):
    MODEL_CODES = [262]
    FRIENDLY_DESC = "Outdoor Air Humidity Sensor"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.temp_mk_current = None
        self.temp_mk_daily_max = None
        self.temp_mk_daily_min = None
        self.temp_trend = None
        self.hum_current = None
        self.hum_daily_max = None
        self.hum_daily_min = None
        self.hum_trend = None

    def _parse_device_specific_status_d_value(self, s):
        """
        Observed example value:
        755(1020/588/1),54(91/24/1),

        Deduced meaning:
        temp[.1F](day-max/day-min/trend?),humidity[%](day-max/day-min/trend?)
        """
        temp_str, hum_str, *_ = s.split(',')
        self.temp_mk_current, self.temp_mk_daily_max, self.temp_mk_daily_min, self.temp_trend = [_temp_to_mk(v) for v in _parse_stats_value(temp_str)]
        self.hum_current, self.hum_daily_max, self.hum_daily_min, self.hum_trend = _parse_stats_value(hum_str)

    def __str__(self):
        s = super().__str__()
        if self.temp_mk_current:
            s += f": {self.temp_mk_current*1e-3-273.15:.1f}°C / {self.hum_current}%"
        return s


class RainPoint2ZoneTimer(HomgarSubDevice):
    """Legacy 2-Zone Water Timer (paramVersion=2 comma-separated format)."""
    MODEL_CODES = [261]
    FRIENDLY_DESC = "2-Zone Water Timer"

    def _parse_device_specific_status_d_value(self, s):
        """
        TODO deduce meaning of these fields.
        Observed example value:
        0,9,0,0,0,0|0,1291,0,0,0,0

        What we know so far:
        left/right zone separated by '|' character
        fields for each zone: ?,last-usage[.1l],?,?,?,?
        """
        pass


# ---------------------------------------------------------------------------
# Newer-firmware devices: hub HWG023WRF and 2-zone timer HTV213FRF /
# HTV214FRF (model 288) both use the hex-packed TLV status format and do
# NOT emit the legacy `<general>;<specific>` prefix.
# ---------------------------------------------------------------------------


# dpCode (aka typeCode) for known identities on HTV213FRF/HTV214FRF, model
# 288. Per-port statuses come in as records with typeCode==dp_code and
# dp_id matching the port-specific entry in the productModel catalog.
_HTV213_DP_CODES = {
    "STA_RSSI": 32,
    "STA_BAT": 31,
    "STA_WKSTATE": 30,
    "STA_ALARM": 2,
    "STA_EVTIME": 21,
    "STA_DURATION": 19,
    "STA_LASTUSAGE": 15,
}

# Per-port dpIds for the 2-zone timer (model 288). Port 1 = Sprinklers,
# port 2 = Dripline by default (from portDescribe in the API).
_HTV213_PORT_DP_IDS = {
    1: {
        "STA_WKSTATE": 25,
        "STA_ALARM": 29,
        "STA_EVTIME": 33,
        "STA_DURATION": 37,
        "STA_LASTUSAGE": 41,
    },
    2: {
        "STA_WKSTATE": 26,
        "STA_ALARM": 30,
        "STA_EVTIME": 34,
        "STA_DURATION": 38,
        "STA_LASTUSAGE": 42,
    },
}


@dataclass
class ZonePortStatus:
    """Per-zone state for a multi-port irrigation timer."""
    port: int
    wkstate: Optional[int] = None          # bit0=running, other bits flags
    alarm: Optional[int] = None
    ev_time: Optional[int] = None          # device-relative event-start
    duration_s: Optional[int] = None       # requested run duration (seconds)
    last_usage_dl: Optional[int] = None    # last-cycle usage, units 0.1 L

    @property
    def running(self) -> bool:
        return bool(self.wkstate and self.wkstate & 1)


class RainPoint2ZoneTimer_V2(HomgarSubDevice):
    """2-Zone Water Timer on paramVersion>=16 firmware (hex TLV format).

    Models 288 (HTV214FRF / shown as HTV213FRF) and 261 (older HTV213FRF)
    on new firmware. Port 1 is the 'first' hose output, port 2 the second.
    """

    MODEL_CODES = [288]
    FRIENDLY_DESC = "2-Zone Water Timer (v2)"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.battery_state: Optional[int] = None
        self.ports: Dict[int, ZonePortStatus] = {
            1: ZonePortStatus(port=1),
            2: ZonePortStatus(port=2),
        }

    def _parse_device_specific_status_d_value(self, s: str):
        """Newer firmware sends hex-TLV status (``NN#<hex>...``), no ';'
        separator. The base class _parse_status_d_value passes the whole
        value through as ``s`` when there is no ';'."""
        records = parse_tlv_d_value(s)
        for rec in records:
            self._apply_record(rec)

    def _apply_record(self, rec: DpRecord) -> None:
        # Shared (non-port-scoped) codes
        if rec.type_code == _HTV213_DP_CODES["STA_RSSI"]:
            # 2 bytes come through; firmware treats low byte as signed dBm.
            self.rf_rssi = _to_int_le(rec.payload[:1], signed=True)
            return
        if rec.type_code == _HTV213_DP_CODES["STA_BAT"]:
            self.battery_state = _to_int_le(rec.payload, signed=False)
            return

        # Per-port codes — resolve port via dpId lookup
        for port, ids in _HTV213_PORT_DP_IDS.items():
            if rec.dp_id != ids.get(self._identity_for(rec.type_code)):
                continue
            status = self.ports[port]
            ident = self._identity_for(rec.type_code)
            if ident == "STA_WKSTATE":
                status.wkstate = _to_int_le(rec.payload)
            elif ident == "STA_ALARM":
                status.alarm = _to_int_le(rec.payload)
            elif ident == "STA_EVTIME":
                status.ev_time = _to_int_le(rec.payload)
            elif ident == "STA_DURATION":
                status.duration_s = _to_int_le(rec.payload)
            elif ident == "STA_LASTUSAGE":
                status.last_usage_dl = _to_int_le(rec.payload)
            return

    @staticmethod
    def _identity_for(type_code: int) -> Optional[str]:
        for name, code in _HTV213_DP_CODES.items():
            if code == type_code:
                return name
        return None

    def __str__(self) -> str:
        tail = []
        for port, s in self.ports.items():
            state = "RUN" if s.running else "idle"
            tail.append(f"p{port}={state}/d={s.duration_s}s/u={s.last_usage_dl}")
        return f"{super().__str__()} [{', '.join(tail)}]"


class RainPointDisplayHubV2(HomgarHubDevice):
    """Newer irrigation hub (HWG023WRF, model 273).

    The hub itself doesn't carry irrigation state — sub-devices do. This class
    exists mainly so that ``getDeviceByHid`` can produce the right hub type
    and so downstream HA code can identify the hub model.
    """

    MODEL_CODES = [273]
    FRIENDLY_DESC = "Smart+ Irrigation Hub (HWG023WRF)"


MODEL_CODE_MAPPING = {
    code: clazz
    for clazz in (
        RainPointDisplayHub,
        RainPointDisplayHubV2,
        RainPointSoilMoistureSensor,
        RainPointRainSensor,
        RainPointAirSensor,
        RainPoint2ZoneTimer,
        RainPoint2ZoneTimer_V2,
    ) for code in clazz.MODEL_CODES
}
