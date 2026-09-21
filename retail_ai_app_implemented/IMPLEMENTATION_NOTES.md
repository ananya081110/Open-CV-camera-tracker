# Retail AI Command Center – Live Operations Upgrade

This build keeps the existing live camera + FastAPI architecture and upgrades the dashboard into a real-time retail operations command center.

## New dashboard capabilities

- Live Floor with the existing MJPEG AI camera feed
- Live WebSocket state updates
- Live alert feed with acknowledgement
- Zone occupancy and staff coverage
- Customer intelligence and dwell/intent signals
- Customer journey/service-gap view
- Store map with live zone state
- Staff Operations page
- Store Analytics page
- AI Operations Insights page
- Security & Alerts page
- Camera Management
- System Health
- Browser-side alert sound toggle

## New backend capabilities

- `GET /api/v1/state` unified live state
- `GET /api/v1/zones` zone + staff coverage state
- `POST /api/v1/alerts/{alert_id}/ack` alert acknowledgement
- Zone-level staff absence monitoring
- Staff coverage gap events are published into `LIVE_STATE`
- Staff alerts are sent through the existing `RetailNotificationService`
- Configurable confirmation and cooldown windows

## Staff coverage logic

A zone is considered uncovered when:

1. At least one non-staff tracked person is present in the zone.
2. At least one staff tracker ID is configured for the camera system.
3. No configured staff tracker is currently detected in that zone.
4. The condition remains true for `RETAIL_ZONE_STAFF_CONFIRM_SECONDS`.
5. The cooldown has elapsed since the previous alert for that zone.

The system deliberately does **not** assume that an arbitrary detected person is staff.

## Configure staff tracker IDs

The current POC uses persistent tracker IDs. For a local run, you can set them without editing `config.py`:

```bash
export RETAIL_STAFF_IDS=1,4
```

Then run the AI engine.

You can also set:

```bash
export RETAIL_ZONE_STAFF_CONFIRM_SECONDS=8
export RETAIL_ZONE_STAFF_ALERT_COOLDOWN_SECONDS=120
```

If `RETAIL_STAFF_IDS` is empty, the UI will show staff coverage as **unconfigured** and the backend will not send false staff-absence alerts.

## Run from the current project root

Because this implementation is stored as a subfolder of the main project, run the backend from the main project root:

```bash
cd ~/Downloads/ai_camera_tracker
source .venv/bin/activate
export RETAIL_STAFF_IDS=1,4
python3 retail_ai_app_implemented/retail_live_integration/main_retail_store_intelligence.py
```

Run the dashboard in another terminal:

```bash
cd ~/Downloads/ai_camera_tracker/retail_ai_app_implemented/retail_dashboard
npm install
npm run dev
```

Open `http://localhost:5173`.

## Important limitation

The current staff-identification mechanism is tracker-ID based. That is suitable for this POC but is not yet robust enough for production identity. A production version should replace this with a persistent staff identity mechanism such as badge/appearance-based Re-ID or another explicit staff registration workflow.

## Next update: proactive operations + WhatsApp

Added an optional `RetailWhatsAppService` using Twilio's WhatsApp API. It is disabled unless `RETAIL_WHATSAPP_ENABLED=true` and valid Twilio credentials/senders are configured.

New operational automation:
- Coverage-gap WhatsApp alert when customers are present in a zone and no registered staff tracker is present for the confirmation window.
- Persistent-gap escalation after `RETAIL_ZONE_STAFF_ESCALATION_SECONDS`.
- Billing queue alert when customer count reaches `RETAIL_QUEUE_ALERT_THRESHOLD`.
- Per-zone WhatsApp routing through `RETAIL_ZONE_WHATSAPP` JSON, with a default recipient list as fallback.
- Dashboard Notifications page exposes WhatsApp configuration status, message count, current coverage gaps, and notification diagnostics.
- Live API now exposes `/api/v1/notifications` and includes `notification_status` in the live state.

### WhatsApp setup

Twilio/WhatsApp requires a registered sender. For business-initiated WhatsApp notifications outside an active customer-service window, use an approved WhatsApp Content Template and set `TWILIO_WHATSAPP_CONTENT_SID`. For development, Twilio's WhatsApp Sandbox can be used. See Twilio's official WhatsApp quickstart and template documentation.

Never commit `.env` files, Twilio auth tokens, or real staff phone numbers to Git.

## AI Store Manager Recommendations

The command center now includes a rule-based AI Store Manager recommendation layer. It converts live signals into prioritized actions instead of only displaying raw alerts.

Recommendation categories:
- STAFFING: route an associate to a customer zone without registered staff.
- CUSTOMER: assist a high-intent customer when no staff is nearby.
- QUEUE: increase Billing capacity when the queue crosses the configured threshold.
- TRAFFIC: monitor the busiest active zone when no stronger intervention is active.

Each recommendation includes priority, zone, reason, recommended action, confidence, and update time. Recommendations are exposed in the live WebSocket/API state under `recommendations` and are rendered in the AI Store Manager dashboard.

This is intentionally rule-based for deterministic POC behavior. A future version can replace or augment these rules with a learned forecasting/recommendation model while keeping the same dashboard contract.
