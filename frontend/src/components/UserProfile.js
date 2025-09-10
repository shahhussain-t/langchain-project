import React, { useState, useEffect, useCallback } from 'react';
import './UserProfile.css';

function UserProfile({ guestId }) {
  const [profile, setProfile] = useState({ name: '', email: '' });
  const [isEditing, setIsEditing] = useState(false);
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const fetchProfile = useCallback(async () => {
    try {
      const response = await fetch(`http://localhost:8000/profile/${guestId}`);
      if (response.ok) {
        const profileData = await response.json();
        setProfile({
          name: profileData.name || '',
          email: profileData.email || ''
        });
        if (profileData.name) {
          setIsEditing(false);
        } else {
          setIsEditing(true);
        }
      }
    } catch (error) {
      console.error('Error fetching profile:', error);
    }
  }, [guestId]);

  useEffect(() => {
    if (guestId) {
      fetchProfile();
    }
  }, [guestId, fetchProfile]);

  const saveProfile = async () => {
    if (!profile.name.trim()) {
      setMessage('Please enter your name');
      return;
    }

    setIsLoading(true);
    setMessage('');

    try {
      const response = await fetch('http://localhost:8000/profile', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: profile.name.trim(),
          email: profile.email.trim(),
          guest_id: guestId
        }),
      });

      if (response.ok) {
        setMessage('Profile saved successfully!');
        setIsEditing(false);
        setTimeout(() => setMessage(''), 3000);
      } else {
        throw new Error('Failed to save profile');
      }
    } catch (error) {
      console.error('Error saving profile:', error);
      setMessage('Error saving profile. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleInputChange = (field, value) => {
    setProfile(prev => ({ ...prev, [field]: value }));
  };

  const sendEmailReport = async () => {
    if (!profile.email) {
      setMessage('Please add an email address to your profile first.');
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/send-report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: profile.email,
          guest_id: guestId,
          report_type: 'activity_summary'
        }),
      });

      if (response.ok) {
        const data = await response.json();
        if (data.email_sent) {
          setMessage(`✅ Report sent successfully to ${profile.email}!`);
        } else {
          setMessage(`📋 Report generated! ${data.setup_instructions || 'Email configuration needed to send emails.'}`);
        }
      } else {
        throw new Error('Failed to send report');
      }
    } catch (error) {
      console.error('Error sending report:', error);
      setMessage('❌ Error sending report. Please try again.');
    } finally {
      setIsLoading(false);
      setTimeout(() => setMessage(''), 5000);
    }
  };

  return (
    <div className="user-profile">
      <div className="profile-header">
        <h3>👤 User Profile</h3>
        {!isEditing && profile.name && (
          <button 
            className="edit-btn"
            onClick={() => setIsEditing(true)}
          >
            Edit
          </button>
        )}
      </div>

      {isEditing ? (
        <div className="profile-form">
          <div className="form-group">
            <label htmlFor="name">Name *</label>
            <input
              type="text"
              id="name"
              value={profile.name}
              onChange={(e) => handleInputChange('name', e.target.value)}
              placeholder="Enter your name"
              disabled={isLoading}
            />
          </div>

          <div className="form-group">
            <label htmlFor="email">Email (optional)</label>
            <input
              type="email"
              id="email"
              value={profile.email}
              onChange={(e) => handleInputChange('email', e.target.value)}
              placeholder="Enter your email for reports"
              disabled={isLoading}
            />
          </div>

          <div className="form-actions">
            <button 
              className="save-btn"
              onClick={saveProfile}
              disabled={isLoading}
            >
              {isLoading ? 'Saving...' : 'Save Profile'}
            </button>
            {profile.name && (
              <button 
                className="cancel-btn"
                onClick={() => {
                  setIsEditing(false);
                  setMessage('');
                }}
                disabled={isLoading}
              >
                Cancel
              </button>
            )}
          </div>

          {message && (
            <div className={`message ${message.includes('Error') ? 'error' : 'success'}`}>
              {message}
            </div>
          )}
        </div>
      ) : (
        <div className="profile-display">
          {profile.name ? (
            <div className="profile-info">
              <div className="info-item">
                <span className="label">Name:</span>
                <span className="value">{profile.name}</span>
              </div>
              {profile.email && (
                <div className="info-item">
                  <span className="label">Email:</span>
                  <span className="value">{profile.email}</span>
                </div>
              )}
              <div className="profile-actions">
                {profile.email && (
                  <button 
                    className="report-btn"
                    onClick={sendEmailReport}
                    disabled={isLoading}
                  >
                    📧 Send Activity Report
                  </button>
                )}
              </div>
            </div>
          ) : (
            <div className="no-profile">
              <p>No profile information available.</p>
              <button 
                className="setup-btn"
                onClick={() => setIsEditing(true)}
              >
                Set up Profile
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default UserProfile;